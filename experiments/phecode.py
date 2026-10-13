"""Integer PheCode transfer: fixed 1:10 pairs with 5% hard negatives."""

from method.runtime import load_embeddings, load_pairs, print_f1, score_pairs


def score():
    embeddings, curvature = load_embeddings()
    pairs = load_pairs('phecode_test_pairs.npy')
    scores = score_pairs(embeddings, pairs, curvature, squared_form=True)
    return pairs[:, 2], scores


def main():
    print_f1('phecode', *score())


if __name__ == '__main__':
    main()
