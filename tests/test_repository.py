import json
import pickle
import struct
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "sms_spam_collection.csv"
NOTEBOOK = ROOT / "notebooks" / "sms_spam_classification.ipynb"
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"


def test_dataset_shape_and_class_counts():
    data = pd.read_csv(DATASET, encoding="latin-1")
    assert len(data) == 5572
    assert set(data["v1"]) == {"ham", "spam"}
    assert data["v1"].value_counts().to_dict() == {"ham": 4825, "spam": 747}


def test_notebook_is_valid_and_safe():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    text = NOTEBOOK.read_text(encoding="utf-8")
    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) == 14
    assert "ghp_" not in text
    assert "github_pat_" not in text


def test_saved_naive_bayes_pipeline_predicts():
    with (MODELS / "tfidf_vectorizer.pkl").open("rb") as file:
        vectorizer = pickle.load(file)
    with (MODELS / "multinomial_naive_bayes.pkl").open("rb") as file:
        model = pickle.load(file)
    samples = ["Free prize! Call now to claim your reward", "Can we meet after class today?"]
    predictions = model.predict(vectorizer.transform(samples))
    assert predictions.shape == (2,)
    assert set(np.asarray(predictions).tolist()).issubset({0, 1})


def test_expected_output_images_are_valid_png_files():
    expected = {
        "class-distribution.png",
        "message-length-distribution.png",
        "model-accuracy-comparison.png",
        "naive-bayes-confusion-matrix.png",
    }
    assert {path.name for path in OUTPUTS.glob("*.png")} == expected
    for path in OUTPUTS.glob("*.png"):
        with path.open("rb") as file:
            assert file.read(8) == b"\x89PNG\r\n\x1a\n"
            assert file.read(4) == b"\x00\x00\x00\r"
            assert file.read(4) == b"IHDR"
            width, height = struct.unpack(">II", file.read(8))
        assert width >= 450
        assert height >= 400
