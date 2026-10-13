"""Shared offline inference and positive-class F1; no sampling or calibration."""

import csv
import json
from pathlib import Path

import numpy as np
import torch

from .hcoe import HCOE


RESOURCES = Path(__file__).resolve().parents[1] / 'resources' / 'runtime'


def load_embeddings():
    checkpoint = torch.load(RESOURCES / 'best.pt', map_location='cpu', weights_only=True)
    model = HCOE(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['model'])
    model.eval()
    features = torch.from_numpy(np.load(RESOURCES / 'features.npy', allow_pickle=False))
    nodes = json.loads((RESOURCES / 'nodes.json').read_text(encoding='utf-8'))
    embeddings = model.embed(features, nodes).cpu().numpy()
    return embeddings, model.curvature


def load_pairs(filename):
    # Original five columns: query, candidate, label, domain, hop.
    return np.load(RESOURCES / filename, mmap_mode='r', allow_pickle=False)


def load_csv_columns(filename, columns):
    with (RESOURCES / filename).open(newline='', encoding='utf-8') as handle:
        rows = csv.DictReader(handle)
        return np.array([[int(row[key]) for key in columns] for row in rows], dtype=np.int64)


def load_threshold(task):
    settings = json.loads((RESOURCES / 'thresholds.json').read_text(encoding='utf-8'))
    return settings['tasks'][task]['threshold']


def score_pairs(embeddings, pairs, curvature, batch_size=8192, squared_form=False):
    """CPU float64 symmetric Poincare distance, preserving original operation order."""
    vectors = np.asarray(embeddings, dtype=np.float64)
    radial = np.maximum(1 - curvature * np.einsum('ij,ij->i', vectors, vectors), 1e-12)
    scores = np.empty(len(pairs), dtype=np.float64)
    for offset in range(0, len(pairs), batch_size):
        ids = pairs[offset:offset + batch_size, :2]
        delta = vectors[ids[:, 0]] - vectors[ids[:, 1]]
        squared = np.einsum('ij,ij->i', delta, delta)
        if squared_form:
            # PheCode historically used this algebraic ordering.
            factor = curvature * squared / (radial[ids[:, 0]] * radial[ids[:, 1]])
            scores[offset:offset + len(ids)] = -2 * np.arcsinh(np.sqrt(factor)) / np.sqrt(curvature)
        else:
            norm = np.sqrt(squared)
            factor = np.sqrt(curvature) * norm / np.sqrt(radial[ids[:, 0]] * radial[ids[:, 1]])
            scores[offset:offset + len(ids)] = -2 / np.sqrt(curvature) * np.arcsinh(factor)
    return scores


def at_threshold(labels, scores, threshold):
    labels, scores = np.asarray(labels, dtype=bool), np.asarray(scores)
    if labels.ndim != 1 or labels.shape != scores.shape or not labels.size:
        raise ValueError('Expected equally sized, nonempty label and score vectors')
    if not np.isfinite(scores).all() or not np.isfinite(threshold):
        raise ValueError('Non-finite relation scores or threshold')
    predicted = scores >= threshold
    tp = int(np.sum(labels & predicted))
    fp = int(np.sum(~labels & predicted))
    fn = int(np.sum(labels & ~predicted))
    denominator = 2 * tp + fp + fn
    return {'tp': tp, 'fp': fp, 'fn': fn, 'f1': 2 * tp / denominator if denominator else 0.0}


def print_f1(task, labels, scores):
    result = at_threshold(labels, scores, load_threshold(task))
    print(f"{100 * result['f1']:.2f}")
