"""chrF++ aggregation for the FLORES-200 translation lm-eval tasks.

Used from the generated task YAMLs as `aggregation: !function utils.chrf_pp_agg`, paired with
`metric: chrf` so lm-eval collects (reference, prediction) pairs via doc_to_target and this
function computes the corpus score with word_order=2 (chrF++), rather than plain chrF.
"""
import sacrebleu


def _flat(x):
    return x[0] if isinstance(x, (list, tuple)) else x


def chrf_pp_agg(items):
    """items: list of (reference, prediction) pairs (lm-eval `chrf` collection order).
    Returns corpus chrF++ (character order 6, word order 2, beta 2)."""
    refs = [_flat(ref) for ref, pred in items]
    preds = [_flat(pred) for ref, pred in items]
    return sacrebleu.corpus_chrf(preds, [refs], word_order=2).score
