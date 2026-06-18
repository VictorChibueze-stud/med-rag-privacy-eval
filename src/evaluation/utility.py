from typing import Any
import bert_score

class UtilityEvaluator:
    def __init__(self, metric_model: str='roberta-large') -> None:
        self.metric_model = metric_model
        self._scorer = bert_score.BERTScorer(model_type=self.metric_model, lang='en', rescale_with_baseline=False)

    def compute_bertscore(self, references: list[str], candidates: list[str]) -> dict[str, Any]:
        p, r, f1 = self._scorer.score(candidates, references)
        return {'precision': p.mean().item(), 'recall': r.mean().item(), 'f1': f1.mean().item()}
