"""Достаёт document-level эмбеддинги из обученного чекпоинта Harmonic:
финальное состояние h1[-1]+h2[-1]+h3[-1] после прохода по всему тексту
целиком (без нарезки на чанки) -- то самое O(1)-summary состояние из статьи.

Используется и для разового эмбеддинга вопроса (CPU, в ask.py), и для
пакетного прогона по всем документам (embed_docs.py).
"""

import math
from pathlib import Path

import tiktoken
import torch
import torch.nn as nn

VOCAB = 50257
HIDDEN = 128
TAUS = (4.0, 32.0, 128.0)
A_RANGES = ((0.55, 0.88), (0.90, 0.975), (0.992, 0.999))
MAX_EMBED_TOKENS = 4096  # cap per document for speed на CPU

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = ROOT / "data" / "harmonic_runs" / "harmonic_corpus_final.pt"

_HAS_ASSOC_SCAN = hasattr(torch, "associative_scan")


def _scan_combine(x, y):
    return (y[0] * x[0], y[0] * x[1] + y[1])


def parallel_scan(A, b, h_init):
    orig_dtype = b.dtype
    A, b, h = A.float(), b.float(), h_init.float()
    if _HAS_ASSOC_SCAN:
        p, q = torch.associative_scan(_scan_combine, (A, b), dim=1)
    else:
        p, q = A.clone(), b.clone()
        stride = 1
        while stride < p.shape[1]:
            i = torch.arange(stride, p.shape[1], device=A.device)
            j = i - stride
            pi, pj = p[:, i], p[:, j]
            qi, qj = q[:, i], q[:, j]
            p, q = p.clone(), q.clone()
            p[:, i] = pj * pi
            q[:, i] = pi * qj + qi
            stride *= 2
    return (p * h.unsqueeze(1) + q).to(orig_dtype)


class HarmonicLevel(nn.Module):
    def __init__(self, d, inp_dim, cross_dim, tau, a_min, a_max):
        super().__init__()
        self.inp_dim = inp_dim
        self.a_min = a_min
        self.a_range = a_max - a_min
        total = inp_dim + cross_dim
        self.net_A = nn.Sequential(nn.Linear(total, d * 2), nn.SiLU(), nn.Linear(d * 2, d))
        self.net_B = nn.Sequential(nn.Linear(total, d * 2), nn.SiLU(), nn.Linear(d * 2, d))
        self.norm = nn.LayerNorm(d)
        target = 1.0 - 1.0 / tau
        x = max(0.01, min(0.99, (target - a_min) / self.a_range))
        with torch.no_grad():
            self.net_A[-1].bias.fill_(math.log(x / (1.0 - x)))

    def forward(self, x_in, cross, h0):
        inp = cross if self.inp_dim == 0 else torch.cat([x_in, cross], dim=-1)
        A = self.a_min + self.a_range * torch.sigmoid(self.net_A(inp))
        b = (1.0 - A) * torch.tanh(self.net_B(inp))
        return self.norm(parallel_scan(A, b, h0))


class MinHarmonicFair(nn.Module):
    def __init__(self, vocab=VOCAB, d=HIDDEN):
        super().__init__()
        self.d = d
        self.emb = nn.Embedding(vocab, d)
        ar = A_RANGES
        self.l1 = HarmonicLevel(d, d, d, TAUS[0], ar[0][0], ar[0][1])
        self.l2 = HarmonicLevel(d, d, d, TAUS[1], ar[1][0], ar[1][1])
        self.l3 = HarmonicLevel(d, 0, d, TAUS[2], ar[2][0], ar[2][1])
        self.pred12 = nn.Linear(d, d, bias=False)
        self.pred23 = nn.Linear(d, d, bias=False)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward_states(self, x):
        """Как forward(), но возвращает финальные состояния уровней вместо
        логитов -- h1[-1]+h2[-1]+h3[-1], документ-эмбеддинг размерности d."""
        B, _ = x.shape
        X = self.emb(x)
        h0 = torch.zeros(B, self.d, device=x.device, dtype=X.dtype)
        h1 = self.l1(X, X, h0)
        h1_err = X - self.pred12(h1)
        h2 = self.l2(X, h1_err, h0)
        h2_err = h1 - self.pred23(h2)
        h3 = self.l3(X, h2_err, h0)
        return h1[:, -1, :] + h2[:, -1, :] + h3[:, -1, :]


_model = None
_enc = None


def load_model(device: str = "cpu") -> MinHarmonicFair:
    global _model
    if _model is not None:
        return _model
    model = MinHarmonicFair(d=HIDDEN).to(device)
    ck = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state"])
    model.eval()
    _model = model
    return model


def get_encoder():
    global _enc
    if _enc is None:
        _enc = tiktoken.get_encoding("gpt2")
    return _enc


@torch.no_grad()
def embed_text(text: str, model: MinHarmonicFair = None, device: str = "cpu") -> "torch.Tensor":
    model = model or load_model(device)
    enc = get_encoder()
    ids = enc.encode_ordinary(text)[:MAX_EMBED_TOKENS]
    if not ids:
        return torch.zeros(model.d)
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    state = model.forward_states(x)
    return state[0].cpu()
