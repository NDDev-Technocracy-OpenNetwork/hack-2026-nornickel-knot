"""Harmonic (MinHarmonicFair, ровно та архитектура из бумаги/train_fast.py)
дообучение на корпусе документов Норникель-хака вместо wikitext/enwiki8.

Архитектура и Triton-scan скопированы без изменений из
~/harmonic-labs/Harmonic/harmonic_ssm/architecture/train_fast.py --
это код, которым реально считались цифры в crossover/scaling таблицах бумаги,
не экспериментальный v11.

Добавлено сверху (для хакатона, не было в оригинале):
  - load_data() читает наш текстовый корпус вместо wikitext/enwiki8
  - чекпоинт каждые STEP_CKPT_EVERY шагов, один файл на HF, перезаписывается
  - таблица логов: train_loss, val_loss, ppl, bpt, A1/A2/A3 (иерархия таймскейлов)
"""

import math
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB     = 50257
HIDDEN    = int(os.environ.get("HIDDEN_DIM", 128))
TAUS      = (4.0, 32.0, 128.0)
A_RANGES  = ((0.55, 0.88), (0.90, 0.975), (0.992, 0.999))

SEQ_LEN   = int(os.environ.get("SEQ_LEN", 512))
BATCH     = int(os.environ.get("BATCH_SIZE", 32))
STEPS     = int(os.environ.get("STEPS", 30_000))
LR        = 3e-4
WD        = 0.1
GRAD_CLIP = 1.0
LOG_EVERY = 100
SEED      = 42
VAL_STEPS = 20

CORPUS_PATH = os.environ.get("CORPUS_PATH", "harmonic_corpus.txt")
CKPT_DIR    = os.environ.get("CHECKPOINT_DIR", ".")
HF_TOKEN    = os.environ.get("HF_TOKEN", "")
HF_REPO_ID  = None

STEP_CKPT_EVERY   = 5000
STEP_CKPT_HF_NAME = "harmonic_corpus_latest.pt"  # один файл, перезаписывается


# ---------------------------------------------------------------------------
# HF Hub -- один перезаписываемый чекпоинт
# ---------------------------------------------------------------------------

def _silence_hf():
    try:
        from huggingface_hub.utils import disable_progress_bars
        disable_progress_bars()
    except Exception:
        pass
    import logging
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

_silence_hf()


def _hf_repo_id():
    global HF_REPO_ID
    if HF_REPO_ID is None:
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)
        username = api.whoami()["name"]
        repo_id = f"{username}/harmonic-nornikel-corpus"
        api.create_repo(repo_id=repo_id, exist_ok=True, private=True)
        HF_REPO_ID = repo_id
    return HF_REPO_ID


def upload_checkpoint(local_path, filename=STEP_CKPT_HF_NAME):
    if not HF_TOKEN:
        return
    try:
        from huggingface_hub import upload_file
        rid = _hf_repo_id()
        t0 = time.perf_counter()
        upload_file(path_or_fileobj=local_path, path_in_repo=filename,
                    repo_id=rid, repo_type="model", token=HF_TOKEN)
        mb = os.path.getsize(local_path) / 1024 / 1024
        print(f"  HF: {filename} -> {rid}  ({mb:.0f} MB, {time.perf_counter()-t0:.1f}s)", flush=True)
    except Exception as e:
        print(f"  HF upload error: {e}", flush=True)


# ---------------------------------------------------------------------------
# Triton SSM scan -- без изменений из train_fast.py
# ---------------------------------------------------------------------------

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

_SCAN_BLOCK_D = 64
_USE_TRITON   = False

if _HAS_TRITON:
    @triton.jit
    def _ssm_fwd_kernel(A_ptr, b_ptr, h0_ptr, out_ptr, T, D, stride_b, BLOCK_D: tl.constexpr):
        pid_b, pid_d = tl.program_id(0), tl.program_id(1)
        d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = d_offs < D
        base = pid_b * stride_b
        h = tl.load(h0_ptr + pid_b * D + d_offs, mask=mask, other=0.0).to(tl.float32)
        for t in range(T):
            off = base + t * D + d_offs
            a = tl.load(A_ptr + off, mask=mask, other=0.0).to(tl.float32)
            bb = tl.load(b_ptr + off, mask=mask, other=0.0).to(tl.float32)
            h = a * h + bb
            tl.store(out_ptr + off, h, mask=mask)

    @triton.jit
    def _ssm_bwd_kernel(A_ptr, h_prev_ptr, dout_ptr, dA_ptr, db_ptr, dh0_ptr, T, D, stride_b, BLOCK_D: tl.constexpr):
        pid_b, pid_d = tl.program_id(0), tl.program_id(1)
        d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = d_offs < D
        base = pid_b * stride_b
        grad_h = tl.zeros([BLOCK_D], dtype=tl.float32)
        for i in range(T):
            t = T - 1 - i
            off = base + t * D + d_offs
            do = tl.load(dout_ptr + off, mask=mask, other=0.0).to(tl.float32)
            grad_h = grad_h + do
            h_prev = tl.load(h_prev_ptr + off, mask=mask, other=0.0).to(tl.float32)
            a = tl.load(A_ptr + off, mask=mask, other=0.0).to(tl.float32)
            tl.store(dA_ptr + off, grad_h * h_prev, mask=mask)
            tl.store(db_ptr + off, grad_h, mask=mask)
            grad_h = grad_h * a
        tl.store(dh0_ptr + pid_b * D + d_offs, grad_h, mask=mask)

    class _SSMScanFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, A, b, h_init):
            B, T, D = A.shape
            A, b, h_init = A.contiguous(), b.contiguous(), h_init.contiguous()
            out = torch.empty_like(A)
            grid = (B, triton.cdiv(D, _SCAN_BLOCK_D))
            _ssm_fwd_kernel[grid](A, b, h_init, out, T, D, T * D, BLOCK_D=_SCAN_BLOCK_D, num_warps=2, num_stages=2)
            h_prev = torch.cat([h_init.unsqueeze(1), out[:, :-1, :].contiguous()], dim=1)
            ctx.save_for_backward(A, h_prev)
            ctx.shape = (B, T, D)
            return out

        @staticmethod
        def backward(ctx, grad_out):
            A, h_prev = ctx.saved_tensors
            B, T, D = ctx.shape
            grad_out = grad_out.contiguous()
            dA, db = torch.empty_like(A), torch.empty_like(A)
            dh0 = torch.empty(B, D, device=A.device, dtype=A.dtype)
            grid = (B, triton.cdiv(D, _SCAN_BLOCK_D))
            _ssm_bwd_kernel[grid](A, h_prev, grad_out, dA, db, dh0, T, D, T * D, BLOCK_D=_SCAN_BLOCK_D, num_warps=2, num_stages=2)
            return dA, db, dh0


def _test_triton_scan(device):
    B, T, D = 2, 128, 32
    A_np = torch.rand(B, T, D, device=device) * 0.5 + 0.4
    b_np = torch.randn(B, T, D, device=device) * 0.1
    h0 = torch.zeros(B, D, device=device)
    A_r = A_np.clone().float().requires_grad_(True)
    b_r = b_np.clone().float().requires_grad_(True)
    h, outs_r = h0.clone().float(), []
    for t in range(T):
        h = A_r[:, t] * h + b_r[:, t]
        outs_r.append(h)
    out_r = torch.stack(outs_r, dim=1)
    out_r.sum().backward()
    A_t = A_np.clone().float().requires_grad_(True)
    b_t = b_np.clone().float().requires_grad_(True)
    out_t = _SSMScanFn.apply(A_t, b_t, h0.float())
    out_t.sum().backward()
    fwd = torch.allclose(out_r.detach(), out_t.detach(), atol=1e-4)
    bwd = (torch.allclose(A_r.grad, A_t.grad, atol=1e-4) and torch.allclose(b_r.grad, b_t.grad, atol=1e-4))
    return fwd and bwd


def parallel_scan(A, b, h_init):
    orig = b.dtype
    A, b, h = A.float(), b.float(), h_init.float()
    if _USE_TRITON:
        return _SSMScanFn.apply(A, b, h).to(orig)
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
    return (p * h.unsqueeze(1) + q).to(orig)


# ---------------------------------------------------------------------------
# Harmonic -- MinHarmonicFair, без изменений из train_fast.py
# ---------------------------------------------------------------------------

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

    def mean_gate(self, device, dtype):
        """Средний decay gate A на нулевом входе -- для проверки иерархии A1<A2<A3."""
        total = self.net_A[0].in_features
        zero = torch.zeros(1, 4, total, device=device, dtype=dtype)
        with torch.no_grad():
            A = self.a_min + self.a_range * torch.sigmoid(self.net_A(zero))
        return A.mean().item()


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

    def forward(self, x):
        B, _ = x.shape
        X = self.emb(x)
        h0 = torch.zeros(B, self.d, device=x.device, dtype=X.dtype)
        h1 = self.l1(X, X, h0)
        h1_err = X - self.pred12(h1)
        h2 = self.l2(X, h1_err, h0)
        h2_err = h1 - self.pred23(h2)
        h3 = self.l3(X, h2_err, h0)
        return self.head(h1 + h2 + h3)

    def mean_gates(self, device, dtype):
        return self.l1.mean_gate(device, dtype), self.l2.mean_gate(device, dtype), self.l3.mean_gate(device, dtype)


# ---------------------------------------------------------------------------
# Данные -- корпус хакатона вместо wikitext/enwiki8
# ---------------------------------------------------------------------------

def load_data():
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    print(f"Загрузка корпуса: {CORPUS_PATH}", flush=True)
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    tokens = torch.tensor(enc.encode_ordinary(text), dtype=torch.long)
    n_val = max(1000, len(tokens) // 20)
    train, val = tokens[:-n_val], tokens[-n_val:]
    print(f"  корпус {len(text):,} символов -> {len(tokens):,} токенов  "
          f"(train {len(train):,}, val {len(val):,})", flush=True)
    return train, val


def get_batch(tokens, batch, seq_len, device):
    n = len(tokens) - seq_len - 1
    idxs = torch.randint(0, n, (batch,))
    x = torch.stack([tokens[i:i + seq_len] for i in idxs]).to(device)
    y = torch.stack([tokens[i + 1:i + seq_len + 1] for i in idxs]).to(device)
    return x, y


@torch.no_grad()
def val_loss(model, val_tokens, batch, seq_len, device, n_batches=VAL_STEPS):
    model.eval()
    total = 0.0
    for _ in range(n_batches):
        x, y = get_batch(val_tokens, batch, seq_len, device)
        logits = model(x)
        total += F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1)).item()
    model.train()
    return total / n_batches


# ---------------------------------------------------------------------------
# Обучение -- train_one из train_fast.py + step-чекпоинты + ppl/bpt/A1-A3 в логах
# ---------------------------------------------------------------------------

def save_and_upload(model, opt, sched, step, val_l, device):
    path = os.path.join(CKPT_DIR, "harmonic_corpus_step.pt")
    A1, A2, A3 = model.mean_gates(device, next(model.parameters()).dtype)
    torch.save({
        "step": step, "val_loss": val_l, "bpt": val_l / math.log(2), "ppl": math.exp(val_l),
        "A1": A1, "A2": A2, "A3": A3,
        "model_state": model.state_dict(),
        "optimizer_state": opt.state_dict(),
        "scheduler_state": sched.state_dict(),
    }, path)
    upload_checkpoint(path, STEP_CKPT_HF_NAME)


def train_one(model, train_tokens, val_tokens, device, args_steps, args_batch, args_seq):
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args_steps, eta_min=LR / 10)
    torch.manual_seed(SEED)
    random.seed(SEED)

    header = (f"{'step':>7}  {'train':>7}  {'val':>7}  {'ppl':>9}  {'bpt':>6}"
              f"  {'A1':>6}  {'A2':>6}  {'A3':>6}  {'tok/s':>8}")
    print(header, flush=True)
    print("-" * len(header), flush=True)

    t_total = t_window = 0.0
    dtype = next(model.parameters()).dtype

    for step in range(args_steps + 1):
        if step % LOG_EVERY == 0:
            vl = val_loss(model, val_tokens, args_batch, args_seq, device)
            ppl = math.exp(min(vl, 20))
            bpt = vl / math.log(2)
            A1, A2, A3 = model.mean_gates(device, dtype)
            tps = (LOG_EVERY * args_batch * args_seq) / max(t_window, 1e-9) if step > 0 else 0.0
            print(f"{step:>7}  {'—' if step==0 else f'{train_loss_display:.4f}':>7}  {vl:>7.4f}  "
                  f"{ppl:>9.1f}  {bpt:>6.3f}  {A1:>6.3f}  {A2:>6.3f}  {A3:>6.3f}  {tps/1e3:>7.1f}k",
                  flush=True)
            t_window = 0.0

        if step > 0 and step % STEP_CKPT_EVERY == 0:
            save_and_upload(model, opt, sched, step, vl, device)

        if step == args_steps:
            break

        x, y = get_batch(train_tokens, args_batch, args_seq, device)
        t0 = time.perf_counter()
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        train_loss_display = loss.item()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        sched.step()
        dt = time.perf_counter() - t0
        t_total += dt
        t_window += dt

    save_and_upload(model, opt, sched, args_steps, vl, device)
    print(f"\nГотово. Финал: val={vl:.4f}  bpt={vl/math.log(2):.4f}  ppl={math.exp(min(vl,20)):.1f}", flush=True)


def main():
    global _USE_TRITON
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"GPU: {p.name}  VRAM: {p.total_memory // 1024**2} MB  torch: {torch.__version__}", flush=True)

    if _HAS_TRITON and device.type == "cuda":
        try:
            _USE_TRITON = _test_triton_scan(device)
            print(f"Triton SSM scan: {'OK' if _USE_TRITON else 'FAIL -- Hillis-Steele fallback'}", flush=True)
        except Exception as e:
            print(f"Triton SSM scan: FAIL ({e}) -- Hillis-Steele fallback", flush=True)
    else:
        print("Triton SSM scan: недоступен -- Hillis-Steele fallback", flush=True)

    train_tokens, val_tokens = load_data()
    model = MinHarmonicFair(d=HIDDEN).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nHarmonic (MinHarmonicFair)  {n_params:,} params  hidden={HIDDEN}  "
          f"seq={SEQ_LEN}  batch={BATCH}  steps={STEPS}  "
          f"total_tokens={STEPS*BATCH*SEQ_LEN/1e6:.1f}M", flush=True)
    print(f"checkpoint каждые {STEP_CKPT_EVERY} шагов -> HF: {STEP_CKPT_HF_NAME}\n", flush=True)

    train_one(model, train_tokens, val_tokens, device, STEPS, BATCH, SEQ_LEN)


if __name__ == "__main__":
    main()
