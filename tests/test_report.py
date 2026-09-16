import hashlib
import json
from pathlib import Path

import numpy as np

from experiments import save_json
from metrics import probabilities
from report import generate_report, summarize


def make_results(directory):
    protocol = json.loads((Path(__file__).parents[1] / "protocol.json").read_text())
    protocol.update(smoke=True, cnn_seeds=[0])
    save_json(directory / "protocol.json", protocol)
    save_json(directory / "status.json", {"status": "complete"})
    labels = np.arange(18) % 9
    logits = np.zeros((18, 9))
    logits[np.arange(18), (labels + 1) % 9] = 5
    images = np.random.default_rng(0).integers(0, 256, (18, 28, 28, 3), dtype=np.uint8)
    np.savez_compressed(directory / "evaluation_images.npz", images=images, labels=labels)
    for name in ["logistic", "cnn-seed-0"]:
        run = directory / name
        run.mkdir()
        save_json(run / "status.json", {"status": "complete"})
        save_json(run / "training.json", {"training_seconds": 1.0, "parameters": 1, "converged": False,
                                          "iterations": [1], "warnings": [{"category": "ConvergenceWarning", "message": "Avertissement simulé"}]})
        save_json(run / "temperature.json", {"temperature": 2.0})
        np.savez_compressed(run / "predictions.npz", indices=np.arange(18), labels=labels, logits=logits,
                            probabilities=probabilities(logits), calibrated_probabilities=probabilities(logits, 2.0))
        (run / "history.csv").write_text("epoch,train_nll,train_accuracy,selection_nll,selection_accuracy,seconds\n1,2,0.1,2,0.1,1\n")


def test_report_rebuilds_without_models_or_download_and_keeps_raw_files(tmp_path):
    make_results(tmp_path)
    raw = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in tmp_path.rglob("*") if path.is_file()}
    report = generate_report(tmp_path)
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in raw.items())
    text = (report / "README.md").read_text()
    assert "aucun résultat de test" in text
    assert "Avertissement simulé" in text
    assert "cnn-seed-0-calibrated" in text
    assert (report / "coverage.csv").exists()
    assert (report / "cnn-seed-0-top-errors.png").exists()
    assert (report / "logistic-confusion.csv").exists()


def test_incomplete_study_has_visible_missing_runs(tmp_path):
    protocol = json.loads((Path(__file__).parents[1] / "protocol.json").read_text())
    protocol["smoke"] = False
    save_json(tmp_path / "protocol.json", protocol)
    save_json(tmp_path / "status.json", {"status": "failed"})
    output = generate_report(tmp_path)
    text = (output / "README.md").read_text()
    assert "Étude incomplète" in text
    assert text.count("not_started") == 4


def test_summary_does_not_invent_repeats_for_logistic():
    metrics = {key: 0.2 for key in ["accuracy", "macro_f1", "nll", "brier", "ece", "aurc"]}
    states = [{"kind": "logistic", "scores": metrics}]
    states += [{"kind": "cnn_raw", "scores": dict(metrics, accuracy=value)} for value in [0.5, 0.6, 0.7]]
    summary = summarize(states)
    assert summary[0]["n"] == 1
    assert summary[0]["accuracy_std"] is None
    assert np.isclose(summary[1]["accuracy_mean"], 0.6)
    assert np.isclose(summary[1]["accuracy_std"], 0.1)
