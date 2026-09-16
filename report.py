"""Recalculer tableaux et figures depuis les sorties archivées, sans modèle."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from data import CLASS_CODES
from metrics import classification_metrics, probabilities, reliability_bins, risk_coverage, threshold_metrics, top_errors


NAMES = {"logistic": "Régression logistique", "cnn_raw": "CNN brut", "cnn_calibrated": "CNN calibré"}
METRICS = ["accuracy", "macro_f1", "nll", "brier", "ece", "aurc"]


def read_json(path):
    return json.loads(path.read_text())


def write_csv(path, rows, fields=None):
    if fields is None:
        fields = list(rows[0]) if rows else []
    with Path(path).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_states(directory, config):
    states, statuses = [], []
    expected = ["logistic"] + [f"cnn-seed-{seed}" for seed in config["cnn_seeds"]]
    for name in expected:
        run = directory / name
        status = read_json(run / "status.json") if (run / "status.json").exists() else {"status": "not_started"}
        statuses.append({"run": name, "status": status["status"], "error": status.get("error", "")})
        if status["status"] != "complete":
            continue
        with np.load(run / "predictions.npz") as archive:
            logits, labels = archive["logits"], archive["labels"]
            saved_raw, saved_scaled = archive["probabilities"], archive["calibrated_probabilities"]
            np.testing.assert_array_equal(archive["indices"], np.arange(len(labels)))
        training = read_json(run / "training.json")
        seed = None if name == "logistic" else int(name.rsplit("-", 1)[1])
        temperatures = [("logistic", 1.0)] if seed is None else [
            ("cnn_raw", 1.0), ("cnn_calibrated", read_json(run / "temperature.json")["temperature"])
        ]
        for kind, temperature in temperatures:
            probs = probabilities(logits, temperature)
            np.testing.assert_allclose(probs, saved_scaled if kind == "cnn_calibrated" else saved_raw, rtol=1e-12, atol=1e-12)
            scores = classification_metrics(logits, labels, temperature, config["ece_bins"])
            curve = risk_coverage(probs, labels)
            states.append({
                "id": name if kind != "cnn_calibrated" else name + "-calibrated",
                "kind": kind, "seed": seed, "temperature": temperature,
                "training_seconds": training["training_seconds"],
                "parameters": training["parameters"],
                "scores": dict(scores, aurc=curve["aurc"]),
                "probabilities": probs, "labels": labels,
            })
    return states, statuses


def summarize(states):
    rows = []
    for kind in NAMES:
        selected = [state for state in states if state["kind"] == kind]
        row = {"model": kind, "n": len(selected)}
        for metric in METRICS:
            values = [state["scores"][metric] for state in selected]
            row[metric + "_mean"] = float(np.mean(values)) if values else None
            row[metric + "_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        rows.append(row)
    return rows


def plot_reliability(state, output, bins):
    rows = reliability_bins(state["probabilities"], state["labels"], bins)
    write_csv(output / f"{state['id']}-reliability.csv", rows)
    occupied = [row for row in rows if row["count"]]
    fig, axes = plt.subplots(2, 1, figsize=(6, 7), sharex=True, height_ratios=[3, 1])
    axes[0].plot([0, 1], [0, 1], "--", color="gray", label="Calibration parfaite")
    axes[0].plot([row["confidence"] for row in occupied], [row["accuracy"] for row in occupied], "o-", label="Mesures")
    axes[0].set(xlim=(0, 1), ylim=(0, 1), ylabel="Exactitude dans l'intervalle", title=state["id"])
    axes[0].legend()
    centers = [(row["lower"] + row["upper"]) / 2 for row in rows]
    axes[1].bar(centers, [row["count"] for row in rows], width=0.9 / bins)
    axes[1].set(xlabel="Confiance maximale", ylabel="Effectif")
    fig.tight_layout()
    fig.savefig(output / f"{state['id']}-reliability.png", dpi=140)
    plt.close(fig)


def plot_risk(states, output, seed):
    selected = [state for state in states if state["seed"] == seed or state["kind"] == "logistic"]
    fig, ax = plt.subplots(figsize=(7, 5))
    for state in selected:
        curve = risk_coverage(state["probabilities"], state["labels"])
        ax.plot(curve["coverage"], curve["risk"], label=NAMES[state["kind"]], linewidth=1.3)
    ax.set(xlabel="Couverture : fraction d'images acceptées", ylabel="Risque : taux d'erreur sur les images acceptées",
           xlim=(0, 1), ylim=(0, 1), title=f"Classement par confiance — CNN graine {seed}")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / f"risk-coverage-seed-{seed}.png", dpi=150)
    plt.close(fig)


def error_rows(state, other=None):
    probs, labels = state["probabilities"], state["labels"]
    indices = top_errors(probs, labels)
    return [{
        "rank": rank, "index": int(index), "target": int(labels[index]),
        "prediction": int(probs[index].argmax()), "confidence": float(probs[index].max()),
        "other_confidence": float(other["probabilities"][index].max()) if other is not None else None,
    } for rank, index in enumerate(indices, 1)]


def plot_errors(state, other, images, output):
    rows = error_rows(state, other)
    write_csv(output / f"{state['id']}-top-errors.csv", rows,
              ["rank", "index", "target", "prediction", "confidence", "other_confidence"])
    fig, axes = plt.subplots(2, 5, figsize=(15, 8), layout="constrained")
    for ax in axes.flat:
        ax.axis("off")
    for row, ax in zip(rows, axes.flat):
        ax.imshow(images[row["index"]], interpolation="nearest")
        title = f"#{row['index']} : {CLASS_CODES[row['target']]} → {CLASS_CODES[row['prediction']]}\nBrut : {row['confidence']:.6f}"
        if other is not None:
            title += f" ; calibré : {row['other_confidence']:.6f}"
        ax.set_title(title, fontsize=9)
    fig.suptitle(f"{state['id']} — dix erreurs les plus confiantes, classement brut")
    if not rows:
        fig.text(0.5, 0.5, "Aucune erreur observée", ha="center")
    fig.savefig(output / f"{state['id']}-top-errors.png", dpi=130)
    plt.close(fig)


def export_selection(states, output, config):
    coverage_rows, threshold_rows, pair_rows = [], [], []
    for state in states:
        probs, labels = state["probabilities"], state["labels"]
        curve = risk_coverage(probs, labels)
        write_csv(output / f"{state['id']}-risk-coverage.csv", [
            {"accepted": index + 1, "coverage": float(coverage), "risk": float(risk)}
            for index, (coverage, risk) in enumerate(zip(curve["coverage"], curve["risk"]))
        ])
        for coverage in config["coverage_levels"]:
            count = int(np.ceil(coverage * len(labels)))
            coverage_rows.append({"model": state["id"], "requested_coverage": coverage,
                                  "accepted": count, "coverage": count / len(labels),
                                  "risk": float(curve["risk"][count - 1]), "accuracy": float(1 - curve["risk"][count - 1])})
        for threshold in config["thresholds"]:
            threshold_rows.append(dict(model=state["id"], **threshold_metrics(probs, labels, threshold)))
        predicted = probs.argmax(1)
        mask = (predicted != labels) & (probs.max(1) >= 0.9)
        for target in range(9):
            for prediction in range(9):
                if target != prediction:
                    pair_rows.append({"model": state["id"], "target": target, "prediction": prediction,
                                      "count": int(np.sum(mask & (labels == target) & (predicted == prediction)))})
        matrix = confusion_matrix(labels, predicted, labels=np.arange(9))
        np.savetxt(output / f"{state['id']}-confusion.csv", matrix, fmt="%d", delimiter=",")
    write_csv(output / "coverage.csv", coverage_rows)
    write_csv(output / "thresholds.csv", threshold_rows)
    write_csv(output / "high-confidence-error-pairs.csv", pair_rows)


def plot_learning(directory, output, seeds):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for seed in seeds:
        path = directory / f"cnn-seed-{seed}" / "history.csv"
        if not path.exists():
            continue
        with path.open() as file:
            rows = list(csv.DictReader(file))
        epochs = [int(row["epoch"]) for row in rows]
        for ax, metric in zip(axes, ["nll", "accuracy"]):
            color = f"C{seed}"
            ax.plot(epochs, [float(row[f"train_{metric}"]) for row in rows], "--", color=color, label=f"Train, graine {seed}")
            ax.plot(epochs, [float(row[f"selection_{metric}"]) for row in rows], color=color, label=f"Sélection, graine {seed}")
            ax.set(xlabel="Époque", ylabel="NLL" if metric == "nll" else "Exactitude")
            ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "learning-curves.png", dpi=150)
    plt.close(fig)


def generate_report(directory):
    config, status = read_json(directory / "protocol.json"), read_json(directory / "status.json")
    states, statuses = load_states(directory, config)
    output = directory / "report"
    output.mkdir(exist_ok=True)
    write_csv(output / "statuses.csv", statuses)
    title = "Vérification courte — aucun résultat de test" if config["smoke"] else "Résultats PathMNIST"
    lines = [f"# {title}", "", f"Statut de l'étude : **{status['status']}**.", ""]
    if status["status"] != "complete" or any(row["status"] != "complete" for row in statuses):
        lines += ["**Étude incomplète.** Les mesures partielles ne constituent pas une comparaison finale.", ""]
    lines += ["| Exécution | Statut |", "|---|---|"]
    lines += [f"| {row['run']} | {row['status']} |" for row in statuses]
    for row in statuses:
        if row["error"]:
            lines += ["", f"Erreur {row['run']} : {row['error']}"]
    if not states:
        (output / "README.md").write_text("\n".join(lines) + "\n")
        return output
    run_rows = [dict(model=state["id"], kind=state["kind"], seed=state["seed"], temperature=state["temperature"],
                     training_seconds=state["training_seconds"], parameters=state["parameters"], **state["scores"]) for state in states]
    write_csv(output / "runs.csv", run_rows)
    summary = summarize(states)
    write_csv(output / "summary.csv", summary)
    lines += ["", "## Tous les résultats", "", "| Modèle | T | Exactitude | Macro-F1 | NLL | Brier | ECE | AURC |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in run_rows:
        lines.append(f"| {row['model']} | {row['temperature']:.4f} | " + " | ".join(f"{row[key]:.4f}" for key in METRICS) + " |")
    lines += ["", "Les exactitudes et ECE sont exprimées en fractions, pas en pourcentages. "
              "Le Brier somme les neuf classes (plage [0, 2]). Une NLL, un Brier, une ECE "
              "ou une AURC plus faibles sont préférables.", "", "## Synthèse des trois états", "",
              "| État | Répétitions | Exactitude | Macro-F1 | NLL | Brier | ECE | AURC |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary:
        cells = []
        for metric in METRICS:
            mean, std = row[metric + "_mean"], row[metric + "_std"]
            cells.append("—" if mean is None else f"{mean:.4f}" + (f" ± {std:.4f}" if std is not None else ""))
        lines.append(f"| {NAMES[row['model']]} | {row['n']} | " + " | ".join(cells) + " |")
    lines += ["", "Écarts-types d'échantillon sur les graines CNN, pas des intervalles de confiance. "
              "La baseline linéaire a une seule exécution. Le CNN calibré partage les poids "
              "du CNN brut ; ses lignes ne sont pas des modèles entraînés indépendamment.", ""]
    linear = read_json(directory / "logistic" / "training.json")
    lines += [f"Régression logistique : convergence signalée **{linear['converged']}**, "
              f"itérations {linear['iterations']}, durée {linear['training_seconds']:.2f} s.", ""]
    for warning in linear["warnings"]:
        lines += [f"- {warning['category']} : {warning['message']}"]
    differences = []
    for seed in config["cnn_seeds"]:
        pair = {state["kind"]: state for state in states if state["seed"] == seed}
        if len(pair) != 2:
            continue
        differences.append(dict(seed=seed, **{
            metric: pair["cnn_calibrated"]["scores"][metric] - pair["cnn_raw"]["scores"][metric]
            for metric in METRICS
        }))
    write_csv(output / "paired-differences.csv", differences)
    lines += ["", "[Écarts calibré − brut par graine](paired-differences.csv) · "
              "[Mesures non arrondies et durées](runs.csv)", "", "## Calibration", "",
              "La température est ajustée uniquement sur la calibration ; ces diagrammes "
              "portent sur le test officiel, ou sur la sélection dans un essai court. "
              "Les intervalles vides restent sans mesure. Les effectifs sont affichés.", ""]
    for state in states:
        plot_reliability(state, output, config["ece_bins"])
        lines += [f"![{state['id']}]({state['id']}-reliability.png)", ""]
    export_selection(states, output, config)
    lines += ["## Abstention", "", "Risque = taux d'erreur parmi les images acceptées. "
              "Le classement par confiance départage les ex aequo par indice croissant. "
              "L'AURC est la moyenne des risques pour k = 1 à N exemples acceptés.", "",
              "[Couvertures fixées](coverage.csv) · [Seuils fixés](thresholds.csv)", "",
              "Les couvertures fixées décrivent un classement du test ; elles ne choisissent "
              "pas un seuil déployable avec risque garanti. Un seuil choisi dans la démo "
              "reste une exploration du test déjà observé.", ""]
    for seed in config["cnn_seeds"]:
        plot_risk(states, output, seed)
        lines += [f"![Risque-couverture graine {seed}](risk-coverage-seed-{seed}.png)", ""]
    lines += ["## Erreurs à haute confiance", "",
              "Les dix erreurs sont classées automatiquement par confiance brute décroissante. "
              "Leur confiance calibrée est affichée sur les mêmes images. Les tableaux séparés "
              "du CNN calibré utilisent son propre classement. Les codes des classes figurent "
              "dans le README principal.", "",
              "[Toutes les paires de confusion à confiance ≥ 0,9](high-confidence-error-pairs.csv)", ""]
    with np.load(directory / "evaluation_images.npz") as archive:
        images = archive["images"]
    for state in states:
        if state["kind"] == "cnn_calibrated":
            other = next(candidate for candidate in states if candidate["seed"] == state["seed"] and candidate["kind"] == "cnn_raw")
            write_csv(output / f"{state['id']}-top-errors.csv", error_rows(state, other),
                      ["rank", "index", "target", "prediction", "confidence", "other_confidence"])
        else:
            other = next((candidate for candidate in states if candidate["seed"] == state["seed"] and candidate["kind"] == "cnn_calibrated"), None)
            plot_errors(state, other, images, output)
            lines += [f"![{state['id']}]({state['id']}-top-errors.png)", ""]
    plot_learning(directory, output, config["cnn_seeds"])
    lines += ["## Apprentissage", "", "![Courbes CNN](learning-curves.png)", "",
              "Les mesures train sont prises pendant les mises à jour ; celles de sélection "
              "en fin d'époque, avec des poids fixes.", "",
              "Images PathMNIST / MedMNIST, CC BY 4.0. Attribution et transformations "
              "dans la notice à la racine du dépôt.", ""]
    (output / "README.md").write_text("\n".join(lines))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    print(generate_report(args.results_dir))
