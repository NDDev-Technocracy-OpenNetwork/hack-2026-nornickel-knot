"""Сравнение скорости/качества OCR: Tesseract (CPU, текущий подход через
pymupdf4llm) vs EasyOCR (GPU) -- на одном и том же документе, на одной и
той же инфраструктуре (Modal), чтобы сравнение было честным.

Запуск:
  modal run src/graph/benchmark_ocr.py
"""

import pathlib
import time

import modal

app = modal.App("nornikel-ocr-benchmark")
ROOT = pathlib.Path.cwd()
TEST_PDF = ROOT / "yandex" / "Задача 2. Научный клубок" / "Источники информации" / "Журналы" / "Цветные металлы" / "2010" / "CM_05_10.pdf"

cpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("tesseract-ocr", "tesseract-ocr-rus", "tesseract-ocr-eng")
    .pip_install("pymupdf4llm==1.28.0")
    .add_local_file(str(TEST_PDF), "/root/test.pdf")
)

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("easyocr==1.7.2", "pymupdf==1.28.0")
    .add_local_file(str(TEST_PDF), "/root/test.pdf")
)

surya_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "surya-ocr==0.16.6", "pymupdf==1.28.0", "requests",
        "transformers==4.56.1",  # surya 0.16.6 сам требует >=4.56.1; 5.x ломает pad_token_id
    )
    .add_local_file(str(TEST_PDF), "/root/test.pdf")
)


@app.function(image=cpu_image, timeout=600)
def bench_tesseract() -> dict:
    import pymupdf4llm

    t0 = time.time()
    pages = pymupdf4llm.to_markdown("/root/test.pdf", page_chunks=True, show_progress=False)
    elapsed = time.time() - t0
    sample = pages[0]["text"][:500] if pages else ""
    return {"engine": "tesseract_cpu", "elapsed_sec": elapsed, "pages": len(pages), "sample_page0": sample}


@app.function(image=gpu_image, gpu="H100", timeout=600)
def bench_easyocr() -> dict:
    import fitz
    import easyocr

    t0 = time.time()
    reader = easyocr.Reader(["ru", "en"], gpu=True)
    load_time = time.time() - t0

    doc = fitz.open("/root/test.pdf")
    t1 = time.time()
    results = []
    sample_page0 = ""
    for i in range(doc.page_count):
        page = doc[i]
        text_layer = page.get_text("text")
        if len(text_layer.strip()) >= 20:
            results.append(text_layer)
            continue
        pix = page.get_pixmap(dpi=200)
        img_bytes = pix.tobytes("png")
        ocr_result = reader.readtext(img_bytes, detail=0, paragraph=True)
        text = "\n".join(ocr_result)
        results.append(text)
        if i == 0:
            sample_page0 = text[:500]
    elapsed = time.time() - t1

    return {
        "engine": "easyocr_gpu", "model_load_sec": load_time, "elapsed_sec": elapsed,
        "pages": len(results), "sample_page0": sample_page0 or (results[0][:500] if results else ""),
    }


_worker_reader = None


def _ocr_worker_init():
    global _worker_reader
    import easyocr
    _worker_reader = easyocr.Reader(["ru", "en"], gpu=True)


def _ocr_worker_run(img_bytes: bytes) -> str:
    result = _worker_reader.readtext(img_bytes, detail=0, paragraph=True)
    return "\n".join(result)


@app.function(image=gpu_image, gpu="H100", timeout=600)
def bench_easyocr_parallel(n_workers: int = 8) -> dict:
    import multiprocessing as mp
    import fitz
    import easyocr

    # прогреваем -- скачиваем модель ОДИН раз до старта пула, иначе N
    # процессов разом кидаются качать в один и тот же temp.zip и давят
    # друг друга (FileNotFoundError на os.remove чужого temp-файла)
    easyocr.Reader(["ru", "en"], gpu=False)

    doc = fitz.open("/root/test.pdf")
    ocr_images = []
    for i in range(doc.page_count):
        text_layer = doc[i].get_text("text")
        if len(text_layer.strip()) >= 20:
            continue
        pix = doc[i].get_pixmap(dpi=200)
        ocr_images.append(pix.tobytes("png"))

    t0 = time.time()
    ctx = mp.get_context("spawn")  # fork() + CUDA context не дружат
    with ctx.Pool(n_workers, initializer=_ocr_worker_init) as pool:
        pool_ready = time.time() - t0
        results = pool.map(_ocr_worker_run, ocr_images)
    elapsed = time.time() - t0

    return {
        "engine": f"easyocr_parallel_{n_workers}", "elapsed_sec": elapsed,
        "pool_setup_sec": pool_ready, "pages_ocred": len(ocr_images),
        "sample_page0": results[0][:500] if results else "",
    }


@app.function(image=surya_image, gpu="H100", timeout=600)
def bench_surya() -> dict:
    import fitz
    from surya.detection import DetectionPredictor
    from surya.foundation import FoundationPredictor
    from surya.recognition import RecognitionPredictor

    t0 = time.time()
    foundation_predictor = FoundationPredictor()
    det_predictor = DetectionPredictor()
    rec_predictor = RecognitionPredictor(foundation_predictor)
    load_time = time.time() - t0

    doc = fitz.open("/root/test.pdf")
    images, page_indices = [], []
    for i in range(doc.page_count):
        text_layer = doc[i].get_text("text")
        if len(text_layer.strip()) >= 20:
            continue
        pix = doc[i].get_pixmap(dpi=200)
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        images.append(img)
        page_indices.append(i)

    t1 = time.time()
    # настоящий батч -- весь список картинок одним вызовом, не по одной
    predictions = rec_predictor(images, det_predictor=det_predictor) if images else []
    elapsed = time.time() - t1

    sample_page0 = ""
    if predictions:
        lines = [line.text for line in predictions[0].text_lines]
        sample_page0 = "\n".join(lines)[:500]

    return {
        "engine": "surya_gpu", "model_load_sec": load_time, "elapsed_sec": elapsed,
        "pages_ocred": len(images), "sample_page0": sample_page0,
    }


@app.local_entrypoint()
def main():
    print("Запускаю бенчмарки на Modal...")
    print("\n=== Tesseract (CPU) ===")
    r1 = bench_tesseract.remote()
    print(f"Время: {r1['elapsed_sec']:.1f}с, страниц: {r1['pages']}")
    print(f"Образец (стр 0): {r1['sample_page0'][:300]}")

    print("\n=== EasyOCR (H100 GPU, по одной картинке) ===")
    r2 = bench_easyocr.remote()
    print(f"Загрузка модели: {r2['model_load_sec']:.1f}с, обработка: {r2['elapsed_sec']:.1f}с, страниц: {r2['pages']}")
    print(f"Образец (стр 0): {r2['sample_page0'][:300]}")

    print("\n=== EasyOCR (H100 GPU, 8 параллельных инстансов) ===")
    r4 = bench_easyocr_parallel.remote(8)
    print(f"Настройка пула: {r4['pool_setup_sec']:.1f}с, обработка: {r4['elapsed_sec']:.1f}с, страниц в OCR: {r4['pages_ocred']}")
    print(f"Образец (стр 0): {r4['sample_page0'][:300]}")

    print("\n=== Итог ===")
    print(f"Tesseract:                    {r1['elapsed_sec']:.1f}с")
    print(f"EasyOCR (поштучно):           {r2['elapsed_sec']:.1f}с (+{r2['model_load_sec']:.1f}с загрузка)")
    print(f"EasyOCR (8 параллельно):      {r4['elapsed_sec']:.1f}с (страниц: {r4['pages_ocred']})")
