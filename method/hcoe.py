"""HCOE: level projections, two-sided triplet loss and Mobius path aggregation."""

import torch
from torch import nn


def project(x, c):
    eps = 4e-5 if x.dtype == torch.float32 else 1e-10
    max_norm = (1 - eps) / c**0.5
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(1e-15)
    return x * (max_norm / norm).clamp(max=1.0)


def expmap0(x, c):
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(1e-15)
    sqrt_c = c**0.5
    return project(torch.tanh(sqrt_c * norm) * x / (sqrt_c * norm), c)


def mobius_add(x, y, c):
    x2 = (x * x).sum(-1, keepdim=True)
    y2 = (y * y).sum(-1, keepdim=True)
    xy = (x * y).sum(-1, keepdim=True)
    denominator = (1 + 2 * c * xy + c * c * x2 * y2).clamp_min(1e-15)
    return project(((1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y) / denominator, c)


def distance(x, y, c):
    delta = mobius_add(-x, y, c)
    norm = torch.linalg.vector_norm(delta, dim=-1)
    eps = 4e-5 if x.dtype == torch.float32 else 1e-10
    return 2 / c**0.5 * torch.atanh((c**0.5 * norm).clamp(max=1 - eps))


class HCOE(nn.Module):
    def __init__(self, input_dim=768, dim=256, curvature_scale=1.0, margin=5.0):
        super().__init__()
        self.projections = nn.ModuleList([nn.Linear(input_dim, dim) for _ in range(5)])
        self.curvature = curvature_scale / dim
        self.margin = margin

    def at_level(self, features, level):
        return expmap0(self.projections[level - 1](features), self.curvature)

    def triplet_loss(self, parents, children):
        # Each side is [batch, positive + negatives, dimension].
        p, child = parents[:, :1], children[:, :1]
        positive = distance(p, child, self.curvature)
        parent_loss = (
            positive - distance(child, parents[:, 1:], self.curvature) + self.margin
        ).relu().mean()
        child_loss = (
            positive - distance(p, children[:, 1:], self.curvature) + self.margin
        ).relu().mean()
        return parent_loss + child_loss

    @torch.no_grad()
    def embed(self, features, nodes):
        # Preserve CPU float64 inference: path sums approach the ball boundary.
        features = features.cpu().to(dtype=torch.float64)
        device = features.device
        raw = torch.empty(
            (len(nodes), self.projections[0].out_features),
            device=device, dtype=features.dtype,
        )
        aggregated = torch.empty_like(raw)
        for level in range(1, 6):
            ids = torch.tensor([n['id'] for n in nodes if n['level'] == level], device=device)
            layer = self.projections[level - 1]
            projected = torch.nn.functional.linear(
                features[ids], layer.weight.cpu().to(dtype=features.dtype),
                layer.bias.cpu().to(dtype=features.dtype),
            )
            raw[ids] = expmap0(projected, self.curvature)
            if level == 1:
                aggregated[ids] = raw[ids]
            else:
                parents = torch.tensor(
                    [n['parent'] for n in nodes if n['level'] == level], device=device,
                )
                aggregated[ids] = mobius_add(aggregated[parents], raw[ids], self.curvature)
        return aggregated
