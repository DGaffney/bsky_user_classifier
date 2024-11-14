import numpy as np

def compute_dcg(relevances):
    relevances = np.asarray(relevances)
    discounts = np.log2(np.arange(len(relevances)) + 2)
    return np.sum(relevances / discounts)

def compute_ndcg(actual_relevances, predicted_relevances, k=None):
    order = np.argsort(-predicted_relevances)
    actual_relevances = actual_relevances[order]
    if k is not None:
        actual_relevances = actual_relevances[:k]
    dcg = compute_dcg(actual_relevances)
    idcg = compute_dcg(np.sort(actual_relevances)[::-1])
    return dcg / idcg if idcg > 0 else 0
