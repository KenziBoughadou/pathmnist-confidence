"""Explorateur Streamlit des sorties enregistrées, sans entraînement ni inférence."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from data import CLASS_CODES, CLASS_NAMES
from metrics import classification_metrics, risk_coverage, threshold_metrics, top_errors


DEFAULT_RESULTS = Path(__file__).resolve().parent / "results" / "study"


@st.cache_data
def load_results(directory):
    root = Path(directory)
    status = json.loads((root / "status.json").read_text())
    if status["status"] != "complete":
        raise ValueError("Cette étude est incomplète : aucune exploration finale n'est disponible.")
    config = json.loads((root / "protocol.json").read_text())
    with np.load(root / "evaluation_images.npz") as archive:
        images, labels = archive["images"], archive["labels"]
    runs = {}
    for name in ["logistic"] + [f"cnn-seed-{seed}" for seed in config["cnn_seeds"]]:
        with np.load(root / name / "predictions.npz") as archive:
            runs[name] = {key: archive[key] for key in archive.files}
        np.testing.assert_array_equal(runs[name]["labels"], labels)
        runs[name]["temperature"] = 1.0 if name == "logistic" else json.loads((root / name / "temperature.json").read_text())["temperature"]
    return config, images, labels, runs


def percent(value):
    return "Non définie" if value is None else f"{value:.1%}"


def main():
    st.set_page_config(page_title="PathMNIST Confidence", layout="wide")
    st.title("PathMNIST Confidence")
    st.warning("Projet académique. Aucun usage diagnostique ou clinique.")
    st.write("Explorer la différence entre une classe prédite et la confiance qui lui est attribuée.")
    directory = os.environ.get("PATHMNIST_RESULTS", str(DEFAULT_RESULTS))
    try:
        config, images, labels, runs = load_results(directory)
    except (OSError, ValueError, KeyError) as error:
        st.error(f"Résultats indisponibles : {error}")
        st.stop()
    if config["smoke"]:
        st.info("Essai technique sur la sélection : ces sorties ne sont pas des résultats de test.")
    else:
        st.caption(f"{len(labels):,} images du test officiel · résultats enregistrés · aucun calcul du modèle dans l'application")

    model = st.sidebar.selectbox("Modèle", ["CNN", "Régression logistique"])
    seed = st.sidebar.selectbox("Graine CNN", config["cnn_seeds"], disabled=model != "CNN")
    run = runs[f"cnn-seed-{seed}"] if model == "CNN" else runs["logistic"]
    raw, calibrated = run["probabilities"], run["calibrated_probabilities"]
    confidence_source = st.sidebar.radio("Confiance pour l'abstention", ["Calibrée", "Brute"], disabled=model != "CNN")
    chosen = calibrated if model == "CNN" and confidence_source == "Calibrée" else raw
    threshold = st.sidebar.slider("Seuil d'acceptation", 0.0, 1.0, 0.9, 0.01)
    st.sidebar.caption("L'image est acceptée si sa confiance atteint le seuil. Ce réglage explore des résultats déjà observés ; il ne valide pas un seuil clinique.")
    if model == "CNN":
        st.sidebar.write(f"Température apprise : **{run['temperature']:.4f}**")
        st.sidebar.caption("Les classes prédites restent identiques ; les probabilités et l'abstention peuvent changer.")

    example_tab, selection_tab, metrics_tab = st.tabs(["Explorer une image", "Abstention", "Mesures et méthode"])
    with example_tab:
        subset = st.radio("Exemples", ["Toutes les images", "Dix erreurs les plus confiantes"], horizontal=True)
        if subset == "Toutes les images":
            index = int(st.number_input("Indice de l'image", min_value=0, max_value=len(labels) - 1, value=0, step=1))
        else:
            errors = top_errors(raw, labels)
            if len(errors) == 0:
                st.info("Aucune erreur pour ce modèle dans les sorties enregistrées.")
                return
            index = st.selectbox("Erreur classée par confiance brute", errors.tolist())
            st.caption("Classement automatique par confiance brute décroissante, puis indice ; aucun choix manuel des exemples.")
        image_column, details = st.columns([1, 3])
        with image_column:
            st.image(images[index], width=224, caption=f"Image #{index} · 28 × 28 pixels")
        with details:
            prediction = int(raw[index].argmax())
            st.write(f"**Classe réelle :** {CLASS_NAMES[int(labels[index])]}")
            st.write(f"**Prédiction :** {CLASS_NAMES[prediction]}")
            st.write("Prédiction correcte" if prediction == labels[index] else "Prédiction incorrecte")
            columns = st.columns(2 if model == "CNN" else 1)
            columns[0].metric("Confiance brute", percent(float(raw[index].max())))
            if model == "CNN":
                columns[1].metric("Confiance calibrée", percent(float(calibrated[index].max())))
            accepted = bool(chosen[index].max() >= threshold)
            st.info(f"Décision au seuil {threshold:.0%} : {'acceptation' if accepted else 'abstention'}.")
        distribution = {"Brute": raw[index]}
        if model == "CNN":
            distribution["Calibrée"] = calibrated[index]
        st.bar_chart(pd.DataFrame(distribution, index=CLASS_CODES), stack=False)
        st.caption("Les neuf probabilités somment à 1 pour chaque distribution. Elles ne constituent pas un diagnostic.")
        with st.expander("Noms des classes"):
            st.table(pd.DataFrame({"Code": CLASS_CODES, "Classe": CLASS_NAMES}))

    with selection_tab:
        scores = threshold_metrics(chosen, labels, threshold)
        st.write("Mesures sur l'ensemble des images, indépendamment de l'exemple affiché.")
        columns = st.columns(3)
        columns[0].metric("Couverture", percent(scores["coverage"]))
        columns[1].metric("Images acceptées", f"{scores['accepted']} / {len(labels)}")
        columns[2].metric("Exactitude sur les images acceptées", percent(scores["accuracy"]))
        if scores["accepted"] == 0:
            st.info("Aucune image acceptée : exactitude et risque sont indéfinis.")
        curves = {"Brute": risk_coverage(raw, labels)}
        if model == "CNN":
            curves["Calibrée"] = risk_coverage(calibrated, labels)
        frame = pd.DataFrame({name: curve["risk"] for name, curve in curves.items()}, index=curves["Brute"]["coverage"])
        frame.index.name = "Couverture"
        st.line_chart(frame, x_label="Couverture", y_label="Risque : taux d'erreur", height=350)
        st.caption("Les courbes classent les images par confiance ; les ex aequo sont départagés par indice. Elles ne garantissent pas le risque d'un futur déploiement.")
        st.dataframe(pd.DataFrame([
            dict(état=name, **threshold_metrics(probs, labels, threshold))
            for name, probs in ([('Brut', raw), ('Calibré', calibrated)] if model == "CNN" else [('Brut', raw)])
        ]), hide_index=True)

    with metrics_tab:
        metrics_rows = []
        for name, saved in runs.items():
            temperatures = [(name, 1.0)]
            if name.startswith("cnn"):
                temperatures.append((name + " calibré", saved["temperature"]))
            for label, temperature in temperatures:
                metrics_rows.append(dict(modèle=label, **classification_metrics(saved["logits"], labels, temperature, config["ece_bins"])))
        st.dataframe(pd.DataFrame(metrics_rows), hide_index=True)
        st.write("NLL et Brier évaluent les distributions ; l'ECE mesure ici l'écart confiance–exactitude dans quinze intervalles. Une faible ECE ne suffit pas à établir la qualité d'un modèle.")
        st.write("Les poids sont appris sur l'entraînement officiel. La validation officielle est séparée entre sélection du checkpoint et apprentissage de la température. Le test n'intervient qu'après ces étapes.")
        st.caption("Images : PathMNIST / MedMNIST, CC BY 4.0 ; sources et transformations dans THIRD_PARTY_NOTICES.md.")
        st.link_button("Lire le protocole et les limites", "https://github.com/KenziBoughadou/pathmnist-confidence")


if __name__ == "__main__":
    main()
