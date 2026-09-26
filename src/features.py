"""
Step 05: lexical / heuristic feature extraction (Section 6.4, Step 2.2).

Three features per (change, requirement) candidate pair, each in [0, 1]:
  1. Jaccard overlap of lemmatised nouns / technical terms
  2. requirement-type match (same module prefix / section header)
  3. shared named-entity score

These are called "lexical/heuristic," not "structural," because they come
from surface text, not a requirement dependency network (cf. Hein et al. [2]).
"""
import numpy as np


_NLP = None


def get_nlp(spacy_model: str):
    global _NLP
    if _NLP is None:
        import spacy
        try:
            _NLP = spacy.load(spacy_model, exclude=["parser"])
        except Exception:
            try:
                _NLP = spacy.load(spacy_model)
            except OSError as e:
                raise OSError(
                    f"spaCy model '{spacy_model}' not found. Run: "
                    f"python -m spacy download {spacy_model}"
                ) from e
    return _NLP


def _content_terms(doc):
    """Lemmatised nouns, proper nouns, and technical-looking tokens."""
    terms = set()
    for tok in doc:
        if tok.is_stop or tok.is_punct or tok.is_space:
            continue
        if tok.pos_ in ("NOUN", "PROPN") or tok.like_num:
            terms.add(tok.lemma_.lower())
    return terms


def _entities(doc):
    return {ent.text.lower() for ent in doc.ents}


def jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 0.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def extract_features_for_candidates(
    change_texts, requirement_texts, req_types_change, req_types_candidate, spacy_model
):
    """
    All four lists are aligned, one entry per candidate pair:
      change_texts[i], requirement_texts[i] -> text pair
      req_types_change[i] -> "" (a change rationale has no type of its own;
          kept for API symmetry, unused)
      req_types_candidate[i] -> the candidate requirement's req_type field
    Returns dict of three np.ndarray feature columns, each in [0, 1],
    plus the raw shared-entity counts (needed to normalise within a CR later).
    """
    nlp = get_nlp(spacy_model)
    n = len(change_texts)
    jac = np.zeros(n, dtype=np.float32)
    type_match = np.zeros(n, dtype=np.float32)
    entity_count = np.zeros(n, dtype=np.float32)

    # Batch-process unique texts once for speed.
    unique_texts = list(set(change_texts) | set(requirement_texts))
    docs = {t: d for t, d in zip(unique_texts, nlp.pipe(unique_texts, batch_size=64))}

    for i in range(n):
        c_doc = docs[change_texts[i]]
        r_doc = docs[requirement_texts[i]]
        jac[i] = jaccard(_content_terms(c_doc), _content_terms(r_doc))
        entity_count[i] = len(_entities(c_doc) & _entities(r_doc))
        cand_type = (req_types_candidate[i] or "").strip().lower()
        # type_match compares the candidate's own type against any type token
        # mentioned in the change text (best-effort heuristic; skip if no
        # req_type field is available in the dataset -- Section 6.4, Step 2.2).
        if cand_type:
            type_match[i] = float(cand_type in change_texts[i].lower())

    # Normalise shared-entity counts to [0, 1] by the max count observed
    # (Section 6.4, Step 2.2: "divided by the maximum count in the candidate set").
    max_count = entity_count.max() if len(entity_count) else 0.0
    entity_score = entity_count / max_count if max_count > 0 else entity_count

    return {"jaccard": jac, "type_match": type_match, "entity_score": entity_score}


def combine_lexical_score(feature_dict) -> np.ndarray:
    """Simple unweighted mean of the three normalised lexical features."""
    stacked = np.vstack([feature_dict["jaccard"], feature_dict["type_match"], feature_dict["entity_score"]])
    return stacked.mean(axis=0)
