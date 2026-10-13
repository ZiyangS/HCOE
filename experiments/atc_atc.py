"""ATC: two-hop ancestors for query levels 3, 4 and 5."""

from method.runtime import load_embeddings, load_pairs, print_f1, score_pairs


def score():
    embeddings, curvature = load_embeddings()
    pairs = load_pairs('atc_test_pairs.npy')
    scores = score_pairs(embeddings, pairs, curvature)
    return pairs[:, 2], scores


def main():
    print_f1('atc_atc', *score())


if __name__ == '__main__':
    main()
