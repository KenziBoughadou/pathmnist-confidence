import json

import numpy as np
import pytest
from scipy.special import softmax
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import experiments
from data import make_loader, split_validation
from metrics import classification_metrics, fit_temperature, nll, probabilities, reliability_bins, risk_coverage, threshold_metrics, top_errors
from models import SmallCNN
from training import predict_logits, set_seed, train_epoch


def test_selection_and_calibration_are_disjoint_stratified_and_fixed():
    labels = np.tile(np.arange(9), 100)
    selection, calibration = split_validation(labels)
    assert len(selection) == len(calibration) == 450
    assert not np.intersect1d(selection, calibration).size
    np.testing.assert_array_equal(np.sort(np.r_[selection, calibration]), np.arange(900))
    np.testing.assert_array_equal(np.bincount(labels[selection]), np.full(9, 50))
    for first, second in zip((selection, calibration), split_validation(labels)):
        np.testing.assert_array_equal(first, second)


def test_uniform_predictions_have_known_scores():
    scores = classification_metrics(np.zeros((9, 9)), np.arange(9))
    assert scores["nll"] == pytest.approx(np.log(9))
    assert scores["brier"] == pytest.approx(8 / 9)
    assert scores["accuracy"] == pytest.approx(1 / 9)
    assert scores["ece"] == pytest.approx(0)


def test_ece_bin_edges_and_empty_bins():
    probs = np.array([[0.5, 0.5], [1.0, 0.0]])
    bins = reliability_bins(probs, np.array([0, 1]), bins=2)
    assert bins[0]["count"] == 0 and bins[0]["accuracy"] is None
    assert bins[1] == {"lower": 0.5, "upper": 1.0, "count": 2, "confidence": 0.75, "accuracy": 0.5}


def test_temperature_preserves_argmax_and_reduces_calibration_nll():
    logits = np.tile([8.0, 0.0], (100, 1))
    labels = np.r_[np.zeros(70, dtype=int), np.ones(30, dtype=int)]
    fitted = fit_temperature(logits, labels)
    assert fitted["temperature"] > 1
    assert fitted["calibration_nll_scaled"] < fitted["calibration_nll_raw"]
    np.testing.assert_array_equal(probabilities(logits).argmax(1), probabilities(logits, fitted["temperature"]).argmax(1))
    assert fitted["temperature"] == pytest.approx(8 / np.log(0.7 / 0.3), rel=1e-5)
    raw = classification_metrics(logits, labels)
    scaled = classification_metrics(logits, labels, fitted["temperature"])
    assert raw["accuracy"] == scaled["accuracy"]
    assert raw["macro_f1"] == scaled["macro_f1"]


def test_nll_is_stable_for_extreme_logits():
    logits = np.array([[1000.0, -1000.0], [-1000.0, 1000.0]])
    assert nll(logits, np.array([0, 1])) == pytest.approx(0)
    assert nll(logits, np.array([1, 0])) == pytest.approx(2000)


@pytest.mark.parametrize("temperature", [0, -1, np.nan, np.inf])
def test_invalid_temperature_is_rejected(temperature):
    with pytest.raises(ValueError):
        probabilities(np.zeros((2, 3)), temperature)


def test_risk_coverage_and_thresholds_do_not_hide_errors():
    probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.6, 0.4]])
    labels = np.array([1, 1, 0, 1])
    curve = risk_coverage(probs, labels)
    np.testing.assert_array_equal(curve["order"], np.arange(4))
    np.testing.assert_allclose(curve["risk"], [1, 0.5, 1 / 3, 0.5])
    assert curve["aurc"] == pytest.approx(np.mean([1, 0.5, 1 / 3, 0.5]))
    assert threshold_metrics(probs, labels, 0.95)["accuracy"] is None
    assert threshold_metrics(probs, labels, 0.6)["accepted"] == 4
    np.testing.assert_array_equal(top_errors(probs, labels, 10), [0, 3])


def test_multiclass_temperature_can_change_confidence_ranking():
    logits = np.array([[0.0, -1.0, -10.0], [0.0, -2.0, -2.0]])
    assert np.argmax(probabilities(logits).max(1)) != np.argmax(probabilities(logits, 10).max(1))
    np.testing.assert_array_equal(probabilities(logits).argmax(1), probabilities(logits, 10).argmax(1))


def test_cnn_training_reload_and_reproducibility(tmp_path):
    images = np.random.default_rng(3).integers(0, 256, size=(12, 28, 28, 3), dtype=np.uint8)
    labels = np.arange(12) % 9
    states = []
    for _ in range(2):
        set_seed(7)
        model = SmallCNN()
        initial = {key: value.clone() for key, value in model.state_dict().items()}
        loader = make_loader(images, labels, batch_size=5, shuffle=True, seed=7)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        train_epoch(model, loader, optimizer)
        assert any(not torch.equal(initial[key], value) for key, value in model.state_dict().items())
        states.append({key: value.clone() for key, value in model.state_dict().items()})
    for key in states[0]:
        torch.testing.assert_close(states[0][key], states[1][key], rtol=0, atol=0)
    torch.save(model.state_dict(), tmp_path / "checkpoint.pt")
    restored = experiments.load_cnn(tmp_path)
    loader = make_loader(images, labels, batch_size=5)
    logits, targets = predict_logits(restored, loader)
    assert logits.shape == (12, 9)
    np.testing.assert_array_equal(targets, labels)
    np.testing.assert_array_equal(logits, predict_logits(model, loader)[0])


def test_training_loss_weights_the_last_incomplete_batch():
    logits = torch.tensor([[5.0, 0], [4.0, 0], [3.0, 0], [2.0, 0], [0.0, 7.0]])
    labels = torch.zeros(5, dtype=torch.long)
    model = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.eye(2))
    loader = DataLoader(TensorDataset(logits, labels), batch_size=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0)
    assert train_epoch(model, loader, optimizer)["nll"] == pytest.approx(nn.CrossEntropyLoss()(logits, labels).item())


def test_test_is_only_read_after_training_and_calibration(tmp_path, monkeypatch):
    events = []
    images = np.zeros((180, 28, 28, 3), dtype=np.uint8)
    labels = np.tile(np.arange(9), 20)

    def load(archive, split):
        events.append(split)
        return images, labels

    class Linear:
        def decision_function(self, features):
            return np.zeros((len(features), 9))

    def train_linear(*args):
        events.append("logistic")
        return Linear()

    def calibrate(directory, *args):
        events.append("calibration")
        directory.mkdir()
        experiments.save_json(directory / "temperature.json", {"temperature": 1})

    monkeypatch.setattr(experiments, "environment", lambda: {})
    monkeypatch.setattr(experiments, "download_data", lambda _: None)
    monkeypatch.setattr(experiments, "load_split", load)
    monkeypatch.setattr(experiments, "train_logistic", train_linear)
    monkeypatch.setattr(experiments, "train_cnn", lambda *args: events.append("cnn"))
    monkeypatch.setattr(experiments, "calibrate_cnn", calibrate)
    monkeypatch.setattr(experiments, "load_cnn", lambda _: None)
    monkeypatch.setattr(experiments, "predict_logits", lambda *args: (np.zeros((180, 9)), labels))
    monkeypatch.setattr(experiments, "save_predictions", lambda *args: events.append("evaluation"))
    experiments.run_study(tmp_path / "study", tmp_path)
    assert events == ["train", "val", "logistic"] + ["cnn"] * 3 + ["calibration"] * 3 + ["test"] + ["evaluation"] * 4


def test_failed_attempt_is_saved_and_cannot_be_overwritten(tmp_path, monkeypatch):
    def fail(*args):
        raise RuntimeError("Erreur simulée")

    monkeypatch.setattr(experiments, "environment", lambda: {})
    monkeypatch.setattr(experiments, "download_data", fail)
    output = tmp_path / "study"
    with pytest.raises(RuntimeError):
        experiments.run_study(output, tmp_path)
    assert json.loads((output / "status.json").read_text())["status"] == "failed"
    assert "Erreur simulée" in (output / "error.txt").read_text()
    with pytest.raises(FileExistsError):
        experiments.run_study(output, tmp_path)
