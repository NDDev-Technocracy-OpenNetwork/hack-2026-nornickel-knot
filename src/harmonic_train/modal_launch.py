"""Modal-лаунчер для train_corpus.py на H100.

Бандлит тренировочный скрипт + корпус прямо в образ (23M символов, ~23MB --
дешевле и надёжнее гонять через сеть при каждом рестарте, чем волюм монтировать).

Запуск:
  modal run src/harmonic_train/modal_launch.py
"""

import os
import pathlib

import modal

app = modal.App("harmonic-nornikel-corpus")
ROOT = pathlib.Path(__file__).parent
PROJECT_ROOT = ROOT.parent.parent

TRAIN_SCRIPT = ROOT / "train_corpus.py"
CORPUS_FILE  = PROJECT_ROOT / "data" / "harmonic_corpus.txt"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "triton", "tiktoken", "huggingface_hub", "numpy")
    .add_local_file(str(TRAIN_SCRIPT), remote_path="/root/train_corpus.py")
    .add_local_file(str(CORPUS_FILE), remote_path="/root/harmonic_corpus.txt")
)


@app.function(
    gpu="H100",
    image=image,
    timeout=6 * 3600,
    secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})],
)
def train(
    steps: int = 30_000,
    hidden_dim: int = 128,
    seq_len: int = 512,
    batch_size: int = 32,
    resume_bytes: bytes = b"",
):
    import sys
    sys.path.insert(0, "/root")

    ckpt_dir = "/root/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    os.environ["CHECKPOINT_DIR"] = ckpt_dir
    os.environ["CORPUS_PATH"]    = "/root/harmonic_corpus.txt"
    os.environ["STEPS"]          = str(steps)
    os.environ["HIDDEN_DIM"]     = str(hidden_dim)
    os.environ["SEQ_LEN"]        = str(seq_len)
    os.environ["BATCH_SIZE"]     = str(batch_size)

    if resume_bytes:
        with open(f"{ckpt_dir}/harmonic_corpus_step.pt", "wb") as f:
            f.write(resume_bytes)
        print("Resume checkpoint загружен локально в контейнер.", flush=True)

    import importlib.util
    spec = importlib.util.spec_from_file_location("train_corpus", "/root/train_corpus.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["train_corpus"] = mod
    spec.loader.exec_module(mod)
    mod.main()

    out_path = f"{ckpt_dir}/harmonic_corpus_step.pt"
    return open(out_path, "rb").read() if os.path.exists(out_path) else b""


@app.local_entrypoint()
def main(steps: int = 30_000, hidden_dim: int = 128, seq_len: int = 512, batch_size: int = 32, resume: bool = False):
    out_dir = PROJECT_ROOT / "data" / "harmonic_runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    resume_bytes = b""
    ckpt_path = out_dir / "harmonic_corpus_final.pt"
    if resume and ckpt_path.exists():
        resume_bytes = ckpt_path.read_bytes()
        print(f"Resume с локального чекпоинта: {ckpt_path} ({len(resume_bytes)//1024} KB)", flush=True)

    print(f"Запуск обучения Harmonic на корпусе: steps={steps} hidden={hidden_dim} "
          f"seq={seq_len} batch={batch_size} resume={resume}", flush=True)
    result = train.remote(steps=steps, hidden_dim=hidden_dim, seq_len=seq_len, batch_size=batch_size,
                          resume_bytes=resume_bytes)

    if result:
        out_path = out_dir / "harmonic_corpus_final.pt"
        out_path.write_bytes(result)
        print(f"Финальный чекпоинт сохранён локально: {out_path} ({len(result)//1024} KB)", flush=True)
    else:
        print("Чекпоинт не получен -- смотри логи выше.", flush=True)
