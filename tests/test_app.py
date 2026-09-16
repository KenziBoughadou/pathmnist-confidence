from pathlib import Path

from streamlit.testing.v1 import AppTest

from test_report import make_results


def test_app_explores_saved_results_and_empty_acceptance(tmp_path, monkeypatch):
    make_results(tmp_path)
    monkeypatch.setenv("PATHMNIST_RESULTS", str(tmp_path))
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=30).run()
    assert not app.exception
    assert "Aucun usage diagnostique" in app.warning[0].value
    assert "Essai technique" in app.info[0].value
    app.slider[0].set_value(1.0).run()
    assert not app.exception
    assert any(metric.value == "Non définie" for metric in app.metric)
    app.main.radio[0].set_value("Dix erreurs les plus confiantes").run()
    assert not app.exception
    app.sidebar.selectbox[0].set_value("Régression logistique").run()
    assert not app.exception
    assert app.sidebar.radio[0].disabled


def test_app_missing_results_is_an_explicit_message(tmp_path, monkeypatch):
    monkeypatch.setenv("PATHMNIST_RESULTS", str(tmp_path / "missing"))
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=30).run()
    assert not app.exception
    assert "Résultats indisponibles" in app.error[0].value
