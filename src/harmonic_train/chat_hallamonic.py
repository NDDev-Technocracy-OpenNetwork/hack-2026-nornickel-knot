"""Генерация текста с Omibranch/hallamonic-1b-instruct (реальный чекпоинт из
бумаги: TinyLlama 1.1B с attention -> HarmonicBlock, дообученный, с chat_template).

model.py / harmonic_block.py скопированы без изменений из
~/harmonic-labs/Harmonic/hallamonic/ -- это код, которым чекпоинт обучен,
не переписываю.

Запуск:
  modal run src/harmonic_train/chat_hallamonic.py --prompt "..."
"""

import os
import pathlib

import modal

app = modal.App("hallamonic-chat")
ROOT = pathlib.Path(__file__).parent
HALLAMONIC_SRC = pathlib.Path.home() / "harmonic-labs" / "Harmonic" / "hallamonic"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.0", "transformers>=4.45.0,<5.0.0", "accelerate", "huggingface_hub")
    .add_local_file(str(HALLAMONIC_SRC / "model.py"), remote_path="/root/model.py")
    .add_local_file(str(HALLAMONIC_SRC / "harmonic_block.py"), remote_path="/root/harmonic_block.py")
)


def generate_custom(model, tokenizer, ids, max_new_tokens=120, temperature=0.8, top_p=0.9, device="cuda"):
    """Кастомный loop -- transformers generate() ломает HarmonicBlock через
    DynamicCache (даёт по 1 токену за шаг, контекст теряется)."""
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        for _ in range(max_new_tokens):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(ids, use_cache=False)
            logits = out.logits[:, -1, :].float() / temperature
            probs = F.softmax(logits, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            mask = (cumsum - sorted_probs) > top_p
            sorted_probs[mask] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            next_local = torch.multinomial(sorted_probs, num_samples=1)
            next_id = sorted_idx[0, next_local[0]].reshape(1, 1)
            ids = torch.cat([ids, next_id], dim=1)
            if next_id.item() == tokenizer.eos_token_id:
                break
    return ids


@app.function(gpu="H100", image=image, timeout=900,
              secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})])
def chat(prompt: str, repo_id: str = "Omibranch/hallamonic-1b-instruct", max_new_tokens: int = 120,
         use_chat_template: bool = True):
    import os
    import sys
    import torch

    sys.path.insert(0, "/root")
    import huggingface_hub
    from model import load_hallamonic

    huggingface_hub.login(token=os.environ.get("HF_TOKEN", ""))

    local_dir = "/root/ckpt"
    print(f"Скачиваю {repo_id}...", flush=True)
    huggingface_hub.snapshot_download(repo_id=repo_id, local_dir=local_dir, token=os.environ.get("HF_TOKEN", ""))

    model, tokenizer = load_hallamonic(local_dir, device="cuda")
    model.eval()

    if use_chat_template and tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
        print(f"chat-template ids: {ids.tolist()}", flush=True)
        print(f"decoded prompt: {tokenizer.decode(ids[0])!r}", flush=True)
    else:
        ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")

    prompt_len = ids.shape[1]
    print(f"eos_token_id={tokenizer.eos_token_id}  prompt_len={prompt_len}", flush=True)
    out_ids = generate_custom(model, tokenizer, ids, max_new_tokens=max_new_tokens, device="cuda")
    print(f"generated {out_ids.shape[1] - prompt_len} new tokens, ids: {out_ids[0][prompt_len:prompt_len+15].tolist()}", flush=True)
    full = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    reply = tokenizer.decode(out_ids[0][prompt_len:], skip_special_tokens=False)

    print("\n=== PROMPT ===", flush=True)
    print(prompt, flush=True)
    print("\n=== REPLY ===", flush=True)
    print(reply, flush=True)
    return reply


@app.local_entrypoint()
def main(prompt: str = "Расскажи, что ты умеешь.", max_new_tokens: int = 120,
         repo_id: str = "Omibranch/hallamonic-1b-instruct", use_chat_template: bool = True):
    chat.remote(prompt=prompt, max_new_tokens=max_new_tokens, repo_id=repo_id,
                use_chat_template=use_chat_template)
