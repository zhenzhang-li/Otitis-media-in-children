"""Global/local layer fusion and cumulative CLS readout.

The three-layer default follows the study head. Layer count, hidden dimension,
attention heads, number of output classes and auxiliary weights are configurable.
"""

import torch
from torch import nn


class AttentionPool1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1)
        )

    def forward(self, x):
        return (x * self.score(x).softmax(1)).sum(1)


class LayerFusion(nn.Module):
    def __init__(self, dim, nclasses, nlayers=3, heads=8, dropout=0.1):
        super().__init__()
        if dim % heads:
            raise ValueError("Fusion dimension must be divisible by attention heads.")
        self.num_layers = nlayers
        self.feature_dim = dim
        self.global_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(nlayers)])
        self.local_attn_pools = nn.ModuleList([AttentionPool1D(dim) for _ in range(nlayers)])
        self.local_fusers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(dim * 2), nn.Linear(dim * 2, dim), nn.GELU(), nn.Dropout(dropout)
                )
                for _ in range(nlayers)
            ]
        )
        self.uncertainty_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(dim * 2),
                    nn.Linear(dim * 2, dim // 2),
                    nn.GELU(),
                    nn.Linear(dim // 2, 1),
                )
                for _ in range(nlayers)
            ]
        )
        self.wg = nn.Sequential(nn.LayerNorm(dim * nlayers), nn.Linear(dim * nlayers, nlayers))
        self.wm = nn.Sequential(nn.LayerNorm(dim * nlayers), nn.Linear(dim * nlayers, nlayers))
        self.cross_attn_g = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.cross_attn_m = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.cross_norm_g = nn.LayerNorm(dim)
        self.cross_norm_m = nn.LayerNorm(dim)
        self.fusion_proj = nn.Sequential(
            nn.LayerNorm(dim * 4), nn.Linear(dim * 4, dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(dim, nclasses)

    @staticmethod
    def _renorm_weights(w):
        return w / w.sum(-1, keepdim=True).clamp_min(1e-6)

    def forward(self, x, return_aux=False):
        gs = []
        ms = []
        us = []
        for i in range(self.num_layers):
            g = self.global_norms[i](x[:, i, 0])
            patch = x[:, i, 1:]
            m = self.local_fusers[i](
                torch.cat([patch.mean(1), self.local_attn_pools[i](patch)], -1)
            )
            u = self.uncertainty_heads[i](torch.cat([g, m], -1)).sigmoid()
            gs.append(g)
            ms.append(m)
            us.append(u)
        a = self.wg(torch.cat(gs, -1)).softmax(-1)
        b = self.wm(torch.cat(ms, -1)).softmax(-1)
        u = torch.cat(us, -1)
        a = self._renorm_weights(a * (1 - u))
        b = self._renorm_weights(b * (1 - u))
        g = sum(a[:, i : i + 1] * gs[i] for i in range(self.num_layers))
        m = sum(b[:, i : i + 1] * ms[i] for i in range(self.num_layers))
        g_token = g.unsqueeze(1)
        m_token = m.unsqueeze(1)
        gc, _ = self.cross_attn_g(
            g_token, m_token, m_token, need_weights=True, average_attn_weights=False
        )
        mc, _ = self.cross_attn_m(
            m_token, g_token, g_token, need_weights=True, average_attn_weights=False
        )
        gg = self.cross_norm_g(gc[:, 0] + g)
        mm = self.cross_norm_m(mc[:, 0] + m)
        features = self.fusion_proj(torch.cat([gg, mm, gg * mm, (gg - mm).abs()], -1))
        logits = self.classifier(features)
        if not return_aux:
            return logits
        return logits, {
            "alpha_tilde": a,
            "beta_tilde": b,
            "uncertainty": u,
            "g_star": g,
            "m_star": m,
        }


def loss_with_aux(model, x, y, weights):
    logits, aux = model(x, return_aux=True)
    a = aux["alpha_tilde"].clamp_min(1e-8)
    b = aux["beta_tilde"].clamp_min(1e-8)
    entropy = 0.5 * ((a * a.log()).sum(1).mean() + (b * b.log()).sum(1).mean())
    alignment = (
        1 - nn.functional.cosine_similarity(aux["g_star"].float(), aux["m_star"].float()).mean()
    )
    return (
        nn.functional.cross_entropy(logits, y)
        + weights.get("gate_mean", 0.02) * aux["uncertainty"].mean()
        + weights.get("negative_weight_entropy", 0.01) * entropy
        + weights.get("branch_alignment", 0.02) * alignment
    )
