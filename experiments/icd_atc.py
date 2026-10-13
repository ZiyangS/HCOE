"""ATC -> ICD-9: center each query against all 500 fixed candidate diagnoses."""

import numpy as np

from method.runtime import load_csv_columns, load_embeddings, print_f1, score_pairs


def query_means(embeddings, query_ids, candidate_ids, curvature, batch_size=128):
    # The entire reference vocabulary is used, independently of association labels.
    e = np.asarray(embeddings, dtype=np.float64)
    q, c = np.asarray(query_ids), np.asarray(candidate_ids)
    squared = np.einsum('ij,ij->i', e, e)
    radial = np.maximum(1 - curvature * squared, 1e-12)
    matrix = np.empty((len(q), len(c)), dtype=np.float64)
    for start in range(0, len(q), batch_size):
        ids = q[start:start + batch_size]
        delta2 = squared[ids, None] + squared[c][None, :] - 2 * e[ids] @ e[c].T
        factor = np.sqrt(
            curvature * np.maximum(delta2, 0) / (radial[ids, None] * radial[c][None, :])
        )
        matrix[start:start + len(ids)] = -2 / np.sqrt(curvature) * np.arcsinh(factor)
    return matrix.mean(axis=1)


def score():
    embeddings, curvature = load_embeddings()
    pairs = load_csv_columns('icd_atc_test_pairs.csv', ('query_id', 'candidate_id', 'label'))
    candidates = load_csv_columns('icd_atc_candidates.csv', ('candidate_id',))[:, 0]
    queries, inverse = np.unique(pairs[:, 0], return_inverse=True)
    means = query_means(embeddings, queries, candidates, curvature)
    scores = score_pairs(embeddings, pairs, curvature) - means[inverse]
    return pairs[:, 2], scores


def main():
    print_f1('icd_atc', *score())


if __name__ == '__main__':
    main()
