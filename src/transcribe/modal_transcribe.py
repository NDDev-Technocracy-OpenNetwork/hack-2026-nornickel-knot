"""Транскрибация Q&A-сессии на H100 через faster-whisper (large-v3,
батчинг, float16) -- спортивный интерес: кто быстрее, мы на GPU или
Данила локально на CPU через faster-whisper.

Запуск:
  modal run src/transcribe/modal_transcribe.py
"""

import pathlib
import time

import modal

app = modal.App("nornikel-transcribe")
ROOT = pathlib.Path.cwd()
AUDIO_PATH = pathlib.Path(
    "/tmp/claude-1000/-home-suicore-hacathons-nornikel/aa1dd9a2-cd19-445e-a2c9-b14c060c5492"
    "/scratchpad/qa_audio/audio.wav"
)
OUT_PATH = ROOT / "data" / "qa_session_transcript.txt"

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04", add_python="3.11")
    .apt_install("ffmpeg")
    .pip_install("faster-whisper==1.1.0", "huggingface_hub", "requests")
    .add_local_file(str(AUDIO_PATH), "/root/audio.wav")
)


@app.function(image=image, gpu="H100", timeout=1800)
def transcribe() -> str:
    from faster_whisper import WhisperModel, BatchedInferencePipeline

    t0 = time.time()
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    pipeline = BatchedInferencePipeline(model=model)
    print(f"Модель загружена за {time.time()-t0:.1f}с")

    t1 = time.time()
    segments, info = pipeline.transcribe(
        "/root/audio.wav", language="ru", beam_size=5,
        vad_filter=True, batch_size=16,
    )
    lines = []
    for seg in segments:
        ts = f"[{int(seg.start//60):02d}:{int(seg.start%60):02d}]"
        lines.append(f"{ts} {seg.text.strip()}")
    elapsed = time.time() - t1
    print(f"Транскрибация заняла {elapsed:.1f}с, длительность аудио {info.duration:.1f}с, "
          f"скорость {info.duration/elapsed:.1f}x реального времени")
    return "\n".join(lines)


@app.local_entrypoint()
def main():
    t0 = time.time()
    text = transcribe.remote()
    total = time.time() - t0
    OUT_PATH.write_text(text, encoding="utf-8")
    print(f"\nГотово за {total:.1f}с (с учётом заливки образа) -> {OUT_PATH.relative_to(ROOT)}")
