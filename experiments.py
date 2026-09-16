"""Sélection, calibration et test : trois usages séparés des données."""

import argparse
import csv
from datetime import datetime, timezone
import gc
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import signal
import sys
import time
import traceback
import warnings

os.environ["MKL_CBWR"] = "COMPATIBLE"

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
import torch

from data import DATA_MD5, DATA_URL, download_data, flat_features, load_split, make_loader, smoke_indices, split_validation
from metrics import classification_metrics, fit_temperature, nll, probabilities
from models import SmallCNN
from training import predict_logits, set_seed, train_epoch


ROOT = Path(__file__).resolve().parent


def now():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def environment():
    files = ["data.py", "models.py", "training.py", "metrics.py", "experiments.py", "protocol.json"]
    cpu = platform.processor()
    if Path("/proc/cpuinfo").exists():
        cpu = next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name"))
    return {
        "python": platform.python_version(), "platform": platform.platform(), "cpu": cpu,
        "packages": {name: version(name) for name in ["torch", "numpy", "scipy", "scikit-learn", "matplotlib", "streamlit", "pytest"]},
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files},
        "data_url": DATA_URL, "data_md5": DATA_MD5,
    }


def train_logistic(directory, images, labels, selection_images, selection_labels, config):
    directory.mkdir()
    save_json(directory / "status.json", {"status": "training", "started_at": now()})
    start = time.perf_counter()
    features = flat_features(images)
    model = LogisticRegression(**config["logistic"])
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        model.fit(features, labels)
    seconds = time.perf_counter() - start
    del features
    gc.collect()
    recorded = [{"category": warning.category.__name__, "message": str(warning.message)} for warning in captured]
    converged = not any(issubclass(warning.category, ConvergenceWarning) for warning in captured)
    logits = model.decision_function(flat_features(selection_images))
    selection = classification_metrics(logits, selection_labels)
    joblib.dump(model, directory / "checkpoint.joblib")
    save_json(directory / "training.json", {
        "training_seconds": seconds, "iterations": model.n_iter_.tolist(),
        "converged": converged, "warnings": recorded, "selection_metrics": selection,
        "parameters": int(model.coef_.size + model.intercept_.size),
    })
    save_json(directory / "status.json", {"status": "trained", "finished_at": now()})
    print(f"Logistic regression: {seconds:.1f}s, converged={converged}, selection NLL={selection['nll']:.4f}", flush=True)
    return model


def train_cnn(directory, images, labels, selection_images, selection_labels, seed, config):
    directory.mkdir()
    save_json(directory / "status.json", {"status": "training", "seed": seed, "started_at": now()})
    set_seed(seed)
    model = SmallCNN()
    train_loader = make_loader(images, labels, config["batch_size"], shuffle=True, seed=seed)
    selection_loader = make_loader(selection_images, selection_labels, config["batch_size"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    best_nll = float("inf")
    best_epoch = None
    start = time.perf_counter()
    fields = ["epoch", "train_nll", "train_accuracy", "selection_nll", "selection_accuracy", "seconds"]
    with (directory / "history.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for epoch in range(1, config["epochs"] + 1):
            epoch_start = time.perf_counter()
            training = train_epoch(model, train_loader, optimizer)
            logits, targets = predict_logits(model, selection_loader)
            selection_nll = nll(logits, targets)
            selection_accuracy = float(np.mean(logits.argmax(1) == targets))
            writer.writerow({
                "epoch": epoch, "train_nll": training["nll"], "train_accuracy": training["accuracy"],
                "selection_nll": selection_nll, "selection_accuracy": selection_accuracy,
                "seconds": time.perf_counter() - epoch_start,
            })
            file.flush()
            if selection_nll < best_nll:
                best_nll, best_epoch = selection_nll, epoch
                torch.save(model.state_dict(), directory / "checkpoint.partial")
                (directory / "checkpoint.partial").replace(directory / "checkpoint.pt")
            print(f"CNN seed={seed} epoch={epoch:02d} train_nll={training['nll']:.4f} selection_nll={selection_nll:.4f} accuracy={selection_accuracy:.4f}", flush=True)
    save_json(directory / "training.json", {
        "training_seconds": time.perf_counter() - start,
        "best_epoch": best_epoch, "selection_nll": best_nll,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    })
    save_json(directory / "status.json", {"status": "trained", "seed": seed, "finished_at": now()})


def load_cnn(directory):
    model = SmallCNN()
    model.load_state_dict(torch.load(directory / "checkpoint.pt", weights_only=True, map_location="cpu"))
    return model


def calibrate_cnn(directory, images, labels, config):
    model = load_cnn(directory)
    logits, targets = predict_logits(model, make_loader(images, labels, config["batch_size"]))
    start = time.perf_counter()
    calibration = fit_temperature(logits, targets)
    calibration["optimization_seconds"] = time.perf_counter() - start
    calibration["examples"] = len(targets)
    np.savez_compressed(directory / "calibration_logits.npz", logits=logits, labels=targets)
    save_json(directory / "temperature.json", calibration)
    save_json(directory / "status.json", {"status": "calibrated", "finished_at": now()})
    print(f"{directory.name}: T={calibration['temperature']:.4f}, calibration NLL {calibration['calibration_nll_raw']:.4f} → {calibration['calibration_nll_scaled']:.4f}", flush=True)


def save_predictions(directory, logits, labels, temperature, config):
    raw = probabilities(logits)
    scaled = probabilities(logits, temperature)
    assert np.array_equal(raw.argmax(1), scaled.argmax(1))
    np.savez_compressed(directory / "predictions.npz", indices=np.arange(len(labels)), labels=labels,
                        logits=logits, probabilities=raw, calibrated_probabilities=scaled)
    save_json(directory / "metrics.json", {
        "raw": classification_metrics(logits, labels, bins=config["ece_bins"]),
        "calibrated": classification_metrics(logits, labels, temperature, config["ece_bins"]) if directory.name.startswith("cnn") else None,
    })
    save_json(directory / "status.json", {"status": "complete", "finished_at": now()})


def run_study(output, data_dir, smoke=False):
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads((ROOT / "protocol.json").read_text())
    config["smoke"] = smoke
    if smoke:
        config.update(cnn_seeds=[0], epochs=2)
        config["logistic"]["max_iter"] = 10
    save_json(output / "protocol.json", config)
    state = {"status": "preparing", "started_at": now()}
    save_json(output / "status.json", state)
    active = None
    try:
        save_json(output / "environment.json", environment())
        archive = download_data(data_dir)
        train_images, train_labels = load_split(archive, "train")
        val_images, val_labels = load_split(archive, "val")
        selection, calibration = split_validation(val_labels, config["split_seed"])
        train_indices = np.arange(len(train_labels))
        if smoke:
            train_indices = smoke_indices(train_labels, 1024)
            train_images, train_labels = train_images[train_indices], train_labels[train_indices]
            selection = selection[smoke_indices(val_labels[selection], 256)]
            calibration = calibration[smoke_indices(val_labels[calibration], 256)]
        np.savez_compressed(output / "split.npz", train=train_indices, selection=selection, calibration=calibration)
        save_json(output / "partitions.json", {
            "train": len(train_labels), "selection": len(selection), "calibration": len(calibration),
            "selection_counts": np.bincount(val_labels[selection], minlength=9).tolist(),
            "calibration_counts": np.bincount(val_labels[calibration], minlength=9).tolist(),
        })
        state.update(status="training")
        save_json(output / "status.json", state)
        active = output / "logistic"
        logistic = train_logistic(active, train_images, train_labels, val_images[selection], val_labels[selection], config)
        for seed in config["cnn_seeds"]:
            active = output / f"cnn-seed-{seed}"
            train_cnn(active, train_images, train_labels, val_images[selection], val_labels[selection], seed, config)
        del train_images, train_labels
        gc.collect()

        state.update(status="calibrating", training_finished_at=now())
        save_json(output / "status.json", state)
        for seed in config["cnn_seeds"]:
            active = output / f"cnn-seed-{seed}"
            calibrate_cnn(active, val_images[calibration], val_labels[calibration], config)
        state.update(status="evaluating", calibration_finished_at=now())
        save_json(output / "status.json", state)
        if smoke:
            test_images, test_labels = val_images[selection], val_labels[selection]
            state["evaluation_split"] = "selection_smoke"
        else:
            # Premier accès aux observations du test, après sélection et calibration.
            test_images, test_labels = load_split(archive, "test")
            state["evaluation_split"] = "official_test"
        save_json(output / "status.json", state)
        np.savez_compressed(output / "evaluation_images.npz", images=test_images, labels=test_labels)
        active = output / "logistic"
        save_predictions(active, logistic.decision_function(flat_features(test_images)), test_labels, 1.0, config)
        for seed in config["cnn_seeds"]:
            active = output / f"cnn-seed-{seed}"
            logits, targets = predict_logits(load_cnn(active), make_loader(test_images, test_labels, config["batch_size"]))
            temperature = json.loads((active / "temperature.json").read_text())["temperature"]
            save_predictions(active, logits, targets, temperature, config)
        state.update(status="complete", finished_at=now())
        save_json(output / "status.json", state)
        print(f"Étude terminée : {output}", flush=True)
    except (Exception, KeyboardInterrupt) as error:
        state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                     error=f"{type(error).__name__}: {error}", finished_at=now())
        save_json(output / "status.json", state)
        (output / "error.txt").write_text(traceback.format_exc())
        if active is not None and active.exists():
            save_json(active / "status.json", state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.output_dir is None:
        parent = Path("results/smoke" if args.smoke else "results")
        args.output_dir = parent / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")

    signal.signal(signal.SIGTERM, interrupt)
    set_seed(0)
    with threadpool_limits(limits=2):
        run_study(args.output_dir, args.data_dir, args.smoke)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
