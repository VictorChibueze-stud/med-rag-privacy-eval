"""Tests for the linear-probe embedding inversion attack."""

import numpy as np

from src.evaluation.inversion_probe import LinearProbeInversion


def test_linear_probe_reconstructs_known_tokens() -> None:
    """A fitted probe returns vocabulary tokens and a bounded ROUGE-L score."""
    texts = [
        "patient reports fever cough",
        "doctor recommends rest hydration",
        "patient has rash itching",
        "blood pressure is elevated",
        "nausea improves with fluids",
        "joint pain needs follow up",
    ]
    embeddings = np.eye(len(texts), dtype=np.float32)

    probe = LinearProbeInversion(max_features=30, top_k_tokens=4)
    probe.fit(texts, embeddings)

    reconstructed = probe.reconstruct(embeddings[0])
    reconstructed_tokens = reconstructed.split()
    vocab = set(probe.vectorizer.get_feature_names_out())

    assert 1 <= len(reconstructed_tokens) <= 4
    assert set(reconstructed_tokens).issubset(vocab)

    score = probe.score(embeddings[0], texts[0])
    assert isinstance(score["reconstructed_text"], str)
    assert 0.0 <= float(score["rouge_l_fmeasure"]) <= 1.0


def test_linear_probe_validation() -> None:
    """The probe validates fit and reconstruction inputs."""
    probe = LinearProbeInversion(max_features=10, top_k_tokens=3)

    with np.testing.assert_raises(RuntimeError):
        probe.reconstruct(np.zeros(4, dtype=np.float32))

    texts = ["alpha beta", "gamma delta"]
    embeddings = np.eye(2, 4, dtype=np.float32)

    with np.testing.assert_raises(ValueError):
        probe.fit(texts[:1], embeddings)

    probe.fit(texts, embeddings)

    with np.testing.assert_raises(ValueError):
        probe.reconstruct(np.zeros((2, 4), dtype=np.float32))

    with np.testing.assert_raises(ValueError):
        probe.reconstruct(np.zeros(3, dtype=np.float32))
