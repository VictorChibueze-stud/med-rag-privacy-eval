"""Load ChatDoctor and related text corpora for RAG and membership inference."""

import json
import random
from pathlib import Path

from sklearn.model_selection import train_test_split

# Default filename when ``data_path`` is a directory.
_CHATDOCTOR_FILENAME = "chatdoctor.json"


def _default_chatdoctor_path(data_path: str) -> Path:
    """Resolve the ChatDoctor JSON path from ``data_path`` (file or parent dir)."""
    p = Path(data_path)
    if p.suffix == ".json":
        return p
    return p / _CHATDOCTOR_FILENAME


def _generate_mock_chatdoctor_corpus(n: int = 1000, seed: int = 42) -> list[str]:
    """Synthesize a medical Q&A-style corpus for experiments without a real file.

    Args:
        n: Number of lines to generate.
        seed: Random seed for reproducible string variation.

    Returns:
        A list of dialogue strings, e.g. "Patient: ... Doctor: ...".
    """
    rng = random.Random(seed)
    symptoms = [
        "headache",
        "cough",
        "fever",
        "sore throat",
        "chest pain",
        "dizziness",
        "fatigue",
        "nausea",
        "rash",
        "joint pain",
    ]
    advices = [
        "Get rest and stay hydrated.",
        "Take a mild over-the-counter analgesic if appropriate.",
        "See a physician if symptoms persist beyond 3 days.",
        "Avoid self-medicating; consult a clinician for a formal diagnosis.",
        "Monitor your temperature and seek urgent care if it spikes.",
    ]
    lines: list[str] = []
    for i in range(n):
        s = symptoms[rng.randrange(0, len(symptoms))]
        a = advices[rng.randrange(0, len(advices))]
        lines.append(
            f"Patient: I have a {s} (case #{i}). "
            f"Doctor: {a} Follow up as needed for your {s}."
        )
    return lines


def _read_strings_from_json(path: Path) -> list[str]:
    """Load a JSON file into a list of dialog strings (flexible encodings)."""
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, list) and not raw:
        return []
    if isinstance(raw, list) and all(isinstance(x, str) for x in raw):
        return list(raw)
    if isinstance(raw, list) and all(isinstance(x, dict) for x in raw):
        out: list[str] = []
        for row in raw:
            t = row.get("text") or row.get("utterance") or row.get("content")
            if t is not None and isinstance(t, str):
                out.append(t)
            else:
                p = str(row.get("patient", "Patient: [unknown]"))
                d = str(row.get("doctor", "Doctor: [unknown]"))
                out.append(f"Patient: {p} Doctor: {d}")
        return out
    msg = f"Expected list of str or list of objects in {path}"
    raise TypeError(msg)


class ChatDoctorLoader:
    """Load and split data for MIA (member vs. non-member) evaluation.

    Attributes:
        data_path: Directory or file path for ChatDoctor (or mock) data.
    """

    def __init__(self, data_path: str) -> None:
        """Build a loader bound to a dataset on disk (or a mock corpus).

        Args:
            data_path: If this path ends in ``.json``, it is the ChatDoctor file.
                Otherwise it is treated as a directory, and
                ``<data_path>/chatdoctor.json`` is used (same as ``data`` →
                ``data/chatdoctor.json``).
        """
        self.data_path = data_path
        self._json_path = _default_chatdoctor_path(data_path)
        self._cached: list[str] | None = None

    def load_data(self) -> list[str]:
        """Load the corpus, preferring ``data/chatdoctor.json`` when it exists.

        If ``<resolved>/chatdoctor.json`` (see ``_default_chatdoctor_path``) is
        missing, a mock dataset of 1,000 synthetic medical Q&A strings is returned so
        downstream code can run without a download.

        Returns:
            A list of document or utterance strings for downstream RAG and analysis.
        """
        if self._cached is not None:
            return list(self._cached)

        if self._json_path.is_file():
            self._cached = _read_strings_from_json(self._json_path)
        else:
            # Fallback: no real ChatDoctor JSON yet — reproducible mock corpus.
            self._cached = _generate_mock_chatdoctor_corpus(1000, seed=42)
        return list(self._cached)

    def get_mia_splits(self, test_size: float = 0.3) -> tuple[list[str], list[str]]:
        """Split the loaded data into MIA training pools (member vs. non-member).

        Uses a fixed ``random_state=42`` so member/non-member pools are stable across
        runs. With ``test_size=0.3`` this yields a 70% member / 30% non-member split.

        Args:
            test_size: Fraction of rows assigned to the non-member (hold-out) pool.

        Returns:
            ``(member_data, non_member_data)`` where the member side is
            ``(1 - test_size)`` of the loaded data.
        """
        all_rows = self.load_data()
        if not 0.0 < test_size < 1.0:
            msg = "test_size must be in (0, 1) for a proper train/test MIA split."
            raise ValueError(msg)
        member_data, non_member_data = train_test_split(
            all_rows,
            test_size=test_size,
            random_state=42,
            shuffle=True,
        )
        return (member_data, non_member_data)
