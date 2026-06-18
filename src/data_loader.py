import json
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
_CHATDOCTOR_FILENAME = 'chatdoctor.json'

def _default_chatdoctor_path(data_path: str) -> Path:
    p = Path(data_path)
    if p.suffix == '.json':
        return p
    return p / _CHATDOCTOR_FILENAME

def _generate_mock_chatdoctor_corpus(n: int=1000, seed: int=42) -> list[str]:
    rng = random.Random(seed)
    symptoms = ['headache', 'cough', 'fever', 'sore throat', 'chest pain', 'dizziness', 'fatigue', 'nausea', 'rash', 'joint pain']
    advices = ['Get rest and stay hydrated.', 'Take a mild over-the-counter analgesic if appropriate.', 'See a physician if symptoms persist beyond 3 days.', 'Avoid self-medicating; consult a clinician for a formal diagnosis.', 'Monitor your temperature and seek urgent care if it spikes.']
    lines: list[str] = []
    for i in range(n):
        s = symptoms[rng.randrange(0, len(symptoms))]
        a = advices[rng.randrange(0, len(advices))]
        lines.append(f'Patient: I have a {s} (case #{i}). Doctor: {a} Follow up as needed for your {s}.')
    return lines

def _read_strings_from_json(path: Path) -> list[str]:
    with path.open(encoding='utf-8') as f:
        raw = json.load(f)
    if isinstance(raw, list) and (not raw):
        return []
    if isinstance(raw, list) and all((isinstance(x, str) for x in raw)):
        return list(raw)
    if isinstance(raw, list) and all((isinstance(x, dict) for x in raw)):
        out: list[str] = []
        for row in raw:
            t = row.get('text') or row.get('utterance') or row.get('content')
            if t is not None and isinstance(t, str):
                out.append(t)
            elif row.get('input') or row.get('output'):
                instruction = str(row.get('instruction', '')).strip()
                patient_input = str(row.get('input', '')).strip()
                doctor_output = str(row.get('output', '')).strip()
                out.append(f'{instruction} Patient: {patient_input} Doctor: {doctor_output}')
            else:
                p = str(row.get('patient', 'Patient: [unknown]'))
                d = str(row.get('doctor', 'Doctor: [unknown]'))
                out.append(f'Patient: {p} Doctor: {d}')
        return out
    msg = f'Expected list of str or list of objects in {path}'
    raise TypeError(msg)

class ChatDoctorLoader:
    def __init__(self, data_path: str) -> None:
        self.data_path = data_path
        self._json_path = _default_chatdoctor_path(data_path)
        self._cached: list[str] | None = None

    def load_data(self) -> list[str]:
        if self._cached is not None:
            return list(self._cached)
        if self._json_path.is_file():
            self._cached = _read_strings_from_json(self._json_path)
        else:
            self._cached = _generate_mock_chatdoctor_corpus(1000, seed=42)
        return list(self._cached)

    def get_mia_splits(self, test_size: float=0.3) -> tuple[list[str], list[str]]:
        all_rows = self.load_data()
        if not 0.0 < test_size < 1.0:
            msg = 'test_size must be in (0, 1).'
            raise ValueError(msg)
        member_data, non_member_data = train_test_split(all_rows, test_size=test_size, random_state=42, shuffle=True)
        return (member_data, non_member_data)
