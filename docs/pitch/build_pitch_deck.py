from __future__ import annotations

from html import escape
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

OUT_DIR = Path(__file__).resolve().parent
SLIDES_DIR = OUT_DIR / "slides"
PPTX_PATH = OUT_DIR / "nornickel_harmonic_labs_pitch.pptx"
HTML_PATH = OUT_DIR / "deck.html"

BG = "#f7f6f3"
INK = "#121110"
INK_MID = "#3a3733"
INK_SOFT = "#8a8680"
ORANGE = "#e85d04"
ORANGE_DIM = "#fff0e8"
LINE = "#ded8ce"
WHITE = "#ffffff"

slides = [
    {
        "kicker": "Норникель AI Hack · трек «Научный клубок»",
        "title": "Живая карта знаний R&D\nдля горной металлургии",
        "body": "Вопрос на естественном языке — структурированный ответ за 4 секунды, с числами, условиями и ссылкой на страницу первоисточника. Поверх 90-летнего архива компании.",
        "tags": ["команда Harmonic Labs", "собственная нейросеть + граф знаний"],
        "footer": "HARMONIC LABS · 2026",
        "layout": "title",
    },
    {
        "kicker": "Проблема",
        "title": "«Мы тонем в горе информации»",
        "quote": "Когда инженер приходит и спрашивает, какие способы очистки воды подходят для конкретного предприятия — начинается ручной квест по архивам. Теряется память компании, дублируются уже проведённые эксперименты.",
        "quote_src": "— постановка задачи, департамент научно-технического развития «Гипроникеля»",
        "metrics": [("90+", "лет наработок: отчёты, регламенты, статьи, доклады, патенты"), ("4", "языка: русский, английский, китайский, испанский"), ("1800+", "файлов разных форматов: PDF, DOCX, PPTX, XLS — многие без текстового слоя")],
        "footer": "HARMONIC LABS · ПРОБЛЕМА",
        "layout": "quote_metrics",
    },
    {
        "kicker": "Решение",
        "title": "Что мы построили за 48 часов",
        "metrics": [("96%", "корпуса обработано: 1070 из 1114 документов прошли полное извлечение фактов"), ("82 706", "узлов графа знаний — материалы, процессы, оборудование, эксперименты, эксперты"), ("159 532", "связей с числами, единицами, условиями и точной страницей источника"), ("3.9 с", "полный ответ: ретрив по графу 1.8 с + синтез YandexGPT — в требовании «3-5 секунд»"), ("8 + 7", "типов сущностей и связей единой онтологии, RU/EN-нормализация имён (canonical name)"), ("~60 с", "живое пополнение: пользователь загружает свой PDF — документ становится частью графа без перезапуска")],
        "footer": "HARMONIC LABS · РЕЗУЛЬТАТ",
        "layout": "metrics6",
    },
    {
        "kicker": "Архитектура",
        "title": "Пайплайн: от сырых файлов до ответа",
        "stages": [("Парсинг + OCR", "PDF / DOCX / PPTX / XLS. Текстовый слой постранично; сканы — EasyOCR на GPU. Кириллица чинится, шум помечается."), ("Извлечение фактов", "Qwen2.5-7B на H100 (vLLM, continuous batching). 10 GPU параллельно: 23 511 чанков за один прогон, ~47 тыс. токенов/с суммарно."), ("Граф знаний", "Дедупликация узлов по канонич. имени (RU и EN сливаются в один узел). Каждая связь несёт число, единицу, условие, страницу."), ("Гибридный ретрив", "Harmonic-эмбеддинги документов + сущностный поиск по графу. 2-hop раскрытие окрестности. 1.8 с, целиком на CPU."), ("Синтез ответа", "YandexGPT (API из РФ — официальный стек). Отвечает только по фактам графа, каждое утверждение — с источником.")],
        "body": "Экспорт в Neo4j — полный контракт данных (documents / chunks / entities / mentions / relations) для промышленной графовой СУБД.",
        "footer": "HARMONIC LABS · АРХИТЕКТУРА",
        "layout": "pipeline",
    },
    {
        "kicker": "Ключевое отличие",
        "title": "Harmonic — наша собственная нейросеть",
        "quote": "Может, у вас там свои видеокарты под столом, вы дообучите какую-нибудь свою нейронку... Это будет вообще очень круто.",
        "quote_src": "— эксперт трека, Q&A-сессия (в шутку). Мы сделали это всерьёз.",
        "cards": [("Опубликованная архитектура", "Иерархическая SSM-модель (state space model) собственной разработки — научная статья опубликована и замечена независимыми исследователями."), ("Обучена на корпусе задачи", "270 МБ текстов горной металлургии. Перплексия на валидации улучшена с 24.5 до 19.9 за ночь дообучения на H100."), ("Читает документ целиком", "Один проход — один вектор документа. Ретрив всего корпуса на обычном CPU, без GPU и без тяжёлых embedding-моделей в проде.")],
        "footer": "HARMONIC LABS · СОБСТВЕННАЯ МОДЕЛЬ",
        "layout": "quote_cards",
    },
    {
        "kicker": "Достоверность",
        "title": "Система не выдумывает — и это проверяемо",
        "cards": [("Каждый факт — со страницей", "Ответ собирается только из фактов графа. У каждого — файл-источник и диапазон страниц: верифицируемость по первоисточнику, как просили эксперты."), ("Белые пятна — как фича", "«Где мы ещё не копали» — система прямо говорит, каких данных нет, вместо правдоподобной выдумки. На контрольных вопросах жюри честно разделяет: вот что есть, вот чего не хватает."), ("Детектор противоречий", "Механическая сверка чисел между источниками: одна пара сущностей, одна единица измерения, сопоставимые условия. Разные условия эксперимента — не противоречие, и система это различает.")],
        "body": "Честный результат на текущем корпусе: явных числовых противоречий не найдено — редкие пересечения источников по одной точной паре сущностей. Механизм готов и сработает по мере роста базы.",
        "footer": "HARMONIC LABS · ЧЕСТНОСТЬ",
        "layout": "cards3_body",
    },
    {
        "kicker": "Интерфейс",
        "title": "Живой граф, а не консоль",
        "cards": [("Обзор графа как в Obsidian", "Интерактивная карта корпуса: фильтры по 8 типам сущностей, текстовый поиск по узлам, силовая укладка, zoom, pan, перетаскивание узлов. Harmonic находит смысловые затравки по запросу — граф собирается вокруг смысла."), ("Свой документ — в граф за минуту", "Кнопка «Добавить свой PDF»: парсинг, извлечение той же онтологией через YandexGPT, слияние с живым графом без перезапуска, пересчёт Harmonic-эмбеддинга."), ("Карта связей под каждым ответом", "К каждому ответу рисуется подграф использованных фактов — видно, через какие узлы связаны источники, и почему система ответила именно так."), ("Многопараметрические запросы", "Материал + процесс + условия + числа в одном вопросе естественным языком — требование задания закрыто гибридным ретривом по типизированному графу.")],
        "footer": "HARMONIC LABS · ИНТЕРФЕЙС",
        "layout": "cards4",
    },
    {
        "kicker": "Ресурсооптимальность",
        "title": "Считали деньги, а не только качество",
        "cards": [("7B вместо 32B", "Замерили на своих данных: Qwen2.5-7B даёт в 3.8 раза больше валидных фактов в единицу времени, чем 32B. Выбор по метрике, не по моде."), ("Ретрив без GPU", "Harmonic — компактная модель: эмбеддинг документа и поиск по 1114 документам работают на CPU. В проде GPU нужен только на этапе загрузки новых архивов."), ("Официальный стек РФ", "Синтез ответов — YandexGPT по выданным ключам. Открытый стек: vLLM, Qwen, networkx, FastAPI, PyTorch. Всё воспроизводимо, ни одной проприетарной «чёрной коробки».")],
        "metrics": [("10× H100", "параллельное извлечение — те же деньги за GPU-часы, в 10 раз быстрее по времени"), ("1.8 с", "ретрив по графу из 82.7 тыс. узлов, CPU"), ("32K", "токенов контекста хватает: факты графа — сжатое знание, а не сырые страницы")],
        "footer": "HARMONIC LABS · РЕСУРСЫ",
        "layout": "cards_metrics",
    },
    {
        "kicker": "Соответствие заданию",
        "title": "Требования и что дальше",
        "checks": ["Многопараметрические запросы естественным языком", "Нормализация RU/EN, синонимов, аббревиатур (canonical name)", "Верификация: источник + страница у каждого факта", "Ответ 3-5 секунд на графе 82.7 тыс. узлов", "Не-консольный интерфейс: веб, граф-обзор, фильтры", "Механизм противоречий между источниками", "Экспорт в Neo4j (полный контракт данных)", "Живое пополнение базы своими документами"],
        "roadmap": ["Извлечение из схем и рисунков (vision-модель) — таблицы с текстовым слоем уже в графе", "Версионирование фактов при обновлении документов", "Стриминг ответа: первые токены за ~1 с", "Дашборды «белых пятен» по областям знаний", "Семантическая дедупликация экспертов (кириллица/латиница)", "Ролевой доступ — по требованиям заказчика"],
        "footer": "HARMONIC LABS · ЧЕК-ЛИСТ · ROADMAP",
        "layout": "checklist",
    },
    {
        "kicker": "Harmonic Labs",
        "title": "Мостик между прошлым опытом\nи будущими открытиями",
        "body": "Именно так заказчик сформулировал образ результата. Наша система уже сегодня отвечает на реальные вопросы экспертов по 96% корпуса — честно, со ссылками, за 4 секунды. И растёт с каждым загруженным документом.",
        "tags": ["граф: 82 706 узлов / 159 532 связи", "собственная SSM-модель Harmonic", "YandexGPT · vLLM · Qwen · Neo4j"],
        "footer": "HARMONIC LABS — НОРНИКЕЛЬ AI HACK 2026 · СПАСИБО",
        "layout": "title",
    },
]

CSS = f"""
* {{ box-sizing: border-box; margin: 0; padding: 0; -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
:root {{ --bg:{BG}; --ink:{INK}; --mid:{INK_MID}; --soft:{INK_SOFT}; --orange:{ORANGE}; --line:{LINE}; }}
body {{ background: var(--bg); color: var(--ink); font-family: Inter, Arial, sans-serif; }}
.deck {{ width: 100%; }}
.slide {{ width: 100vw; height: 100vh; min-height: 720px; padding: 6.8vh 7.5vw; position: relative; display: flex; flex-direction: column; justify-content: center; page-break-after: always; overflow: hidden; }}
.slide::before {{ content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 7px; background: var(--orange); }}
.kicker {{ font: 600 12px 'JetBrains Mono', ui-monospace, monospace; text-transform: uppercase; letter-spacing: .13em; color: var(--orange); margin-bottom: 18px; display:flex; align-items:center; gap:12px; }}
.kicker::before {{ content:''; width:24px; height:1px; background:var(--orange); }}
h1 {{ font: 800 clamp(42px, 5vw, 72px) Unbounded, Inter, sans-serif; line-height: 1.05; letter-spacing: 0; max-width: 980px; }}
h2 {{ font: 800 clamp(34px, 4vw, 54px) Unbounded, Inter, sans-serif; line-height: 1.08; letter-spacing: 0; margin-bottom: 28px; max-width: 980px; }}
.body {{ font-size: 22px; line-height: 1.5; color: var(--mid); max-width: 860px; margin-top: 22px; }}
.quote {{ font: italic 25px Georgia, serif; line-height:1.48; max-width: 920px; border-left: 3px solid var(--orange); padding-left: 24px; color: var(--ink); }}
.quote small {{ display:block; font: 12px 'JetBrains Mono', ui-monospace, monospace; color: var(--soft); margin-top: 14px; text-transform:uppercase; letter-spacing:.06em; }}
.grid {{ display:grid; gap:14px; margin-top: 28px; }}
.g2 {{ grid-template-columns: repeat(2, minmax(0,1fr)); }} .g3 {{ grid-template-columns: repeat(3, minmax(0,1fr)); }}
.card {{ background:#fff; border:1px solid var(--line); border-radius:8px; padding:22px 22px; min-height:132px; }}
.card h3 {{ font: 700 17px Unbounded, Inter, sans-serif; margin-bottom:10px; line-height:1.25; }}
.card p, .card .cap {{ font-size:15px; line-height:1.48; color:var(--mid); }}
.num {{ font: 800 39px 'JetBrains Mono', ui-monospace, monospace; color:var(--orange); line-height:1; margin-bottom:12px; }}
.tags {{ margin-top: 34px; display:flex; flex-wrap:wrap; gap:8px; }}
.tag {{ font: 600 12px 'JetBrains Mono', ui-monospace, monospace; color:var(--mid); border:1px solid var(--line); border-radius:999px; padding:8px 14px; background:#fff; }}
.pipeline {{ display:grid; grid-template-columns: repeat(5, minmax(0,1fr)); gap:2px; background:var(--line); margin-top:10px; }}
.stage {{ background:var(--bg); padding:24px 18px; min-height:250px; }}
.stage b {{ display:block; font:700 16px Unbounded, Inter, sans-serif; margin-bottom:12px; line-height:1.25; }}
.stage span {{ font-size:13.5px; color:var(--mid); line-height:1.46; }}
.checks {{ display:grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap:8px 18px; margin-top: 8px; }}
.check {{ background:#fff; border:1px solid var(--line); border-radius:8px; padding:12px 14px 12px 38px; position:relative; font-size:14.5px; line-height:1.35; color:var(--mid); min-height:54px; }}
.check::before {{ content:'+'; position:absolute; left:14px; top:11px; color:var(--orange); font:700 16px 'JetBrains Mono', monospace; }}
.check.road::before {{ content:'>'; color:var(--soft); }}
.footer {{ position:absolute; left:7.5vw; right:7.5vw; bottom:4.2vh; display:flex; justify-content:space-between; gap:20px; font:600 11px 'JetBrains Mono', ui-monospace, monospace; color:var(--soft); letter-spacing:.08em; text-transform:uppercase; }}
@media print {{ @page {{ size: 16in 9in; margin: 0; }} .slide {{ width: 16in; height: 9in; min-height: 9in; }} }}
"""


def card_html(title: str, body: str) -> str:
    return f'<div class="card"><h3>{escape(title)}</h3><p>{escape(body)}</p></div>'


def metric_html(num: str, cap: str) -> str:
    return f'<div class="card"><div class="num">{escape(num)}</div><div class="cap">{escape(cap)}</div></div>'


def slide_inner(slide: dict, idx: int) -> str:
    h = [f'<div class="kicker">{escape(slide["kicker"])}</div>']
    title_tag = "h1" if slide.get("layout") == "title" else "h2"
    h.append(f'<{title_tag}>{escape(slide["title"]).replace(chr(10), "<br>")}</{title_tag}>')
    if slide.get("quote"):
        h.append(f'<div class="quote">{escape(slide["quote"])}<small>{escape(slide.get("quote_src", ""))}</small></div>')
    if slide.get("body") and slide.get("layout") not in {"pipeline", "cards3_body"}:
        h.append(f'<div class="body">{escape(slide["body"])}</div>')
    if slide.get("metrics"):
        cls = "g3" if len(slide["metrics"]) != 2 else "g2"
        h.append(f'<div class="grid {cls}">' + "".join(metric_html(*m) for m in slide["metrics"]) + '</div>')
    if slide.get("stages"):
        h.append('<div class="pipeline">' + "".join(f'<div class="stage"><b>{escape(t)}</b><span>{escape(b)}</span></div>' for t, b in slide["stages"]) + '</div>')
        h.append(f'<div class="body" style="font-size:18px; max-width:920px;">{escape(slide["body"])}</div>')
    if slide.get("cards"):
        cls = "g2" if len(slide["cards"]) == 4 else "g3"
        h.append(f'<div class="grid {cls}">' + "".join(card_html(*c) for c in slide["cards"]) + '</div>')
    if slide.get("layout") == "cards3_body":
        h.append(f'<div class="body" style="font-size:18px; max-width:980px;">{escape(slide["body"])}</div>')
    if slide.get("checks"):
        checks = [f'<div class="check">{escape(x)}</div>' for x in slide["checks"]]
        roads = [f'<div class="check road">{escape(x)}</div>' for x in slide["roadmap"]]
        h.append('<div class="checks">' + "".join(checks + roads) + '</div>')
    if slide.get("tags"):
        h.append('<div class="tags">' + "".join(f'<span class="tag">{escape(t)}</span>' for t in slide["tags"]) + '</div>')
    h.append(f'<div class="footer"><span>HARMONIC LABS</span><span>{escape(slide["footer"])}</span></div>')
    return "\n".join(h)


def write_html() -> None:
    SLIDES_DIR.mkdir(parents=True, exist_ok=True)
    deck_sections = []
    for i, slide in enumerate(slides, 1):
        inner = slide_inner(slide, i)
        deck_sections.append(f'<section class="slide" id="slide-{i:02d}">\n{inner}\n</section>')
        single = f'<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Harmonic Labs Pitch — slide {i:02d}</title><style>{CSS}</style></head><body><main class="deck">{deck_sections[-1]}</main></body></html>'
        (SLIDES_DIR / f'slide_{i:02d}.html').write_text(single, encoding='utf-8')
    deck = '<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Harmonic Labs — Nornickel Pitch Deck</title><style>' + CSS + '</style></head><body><main class="deck">' + "\n".join(deck_sections) + '</main></body></html>'
    HTML_PATH.write_text(deck, encoding='utf-8')


def rgb(hex_color: str) -> RGBColor:
    hex_color = hex_color.lstrip('#')
    return RGBColor(int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16))


def set_text(tf, text: str, size: int, color: str = INK, bold: bool = False, font: str = 'Aptos'):
    tf.clear()
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(size)
    p.font.color.rgb = rgb(color)
    p.font.bold = bold
    p.font.name = font
    return p


def add_text(slide, x, y, w, h, text, size=18, color=INK, bold=False, font='Aptos', align=None):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.02)
    tf.margin_right = Inches(0.02)
    tf.margin_top = Inches(0.02)
    tf.margin_bottom = Inches(0.02)
    p = set_text(tf, text, size, color, bold, font)
    if align:
        p.alignment = align
    return box


def add_kicker(slide, text):
    add_text(slide, 0.85, 0.48, 8.6, 0.32, text.upper(), 9, ORANGE, True, 'Aptos Mono')
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.62), Inches(0.62), Inches(0.18), Inches(0.015))
    line.fill.solid(); line.fill.fore_color.rgb = rgb(ORANGE); line.line.fill.background()


def add_footer(slide, footer, num):
    add_text(slide, 0.78, 7.12, 5.2, 0.25, 'HARMONIC LABS', 8, INK_SOFT, True, 'Aptos Mono')
    add_text(slide, 6.4, 7.12, 6.0, 0.25, f'{footer} · {num:02d}', 8, INK_SOFT, True, 'Aptos Mono', PP_ALIGN.RIGHT)


def add_card(slide, x, y, w, h, title, body=None, num=None):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid(); shape.fill.fore_color.rgb = rgb(WHITE)
    shape.line.color.rgb = rgb(LINE)
    if num:
        add_text(slide, x + 0.16, y + 0.16, w - 0.32, 0.42, num, 23, ORANGE, True, 'Aptos Mono')
        add_text(slide, x + 0.16, y + 0.70, w - 0.32, h - 0.82, title, 10, INK_MID)
    else:
        add_text(slide, x + 0.16, y + 0.16, w - 0.32, 0.38, title, 12, INK, True)
        add_text(slide, x + 0.16, y + 0.62, w - 0.32, h - 0.72, body or '', 9, INK_MID)


def add_tags(slide, tags, x=0.82, y=5.25):
    cx = x
    for tag in tags:
        w = min(3.5, max(1.4, len(tag) * 0.055 + 0.5))
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(cx), Inches(y), Inches(w), Inches(0.38))
        shape.fill.solid(); shape.fill.fore_color.rgb = rgb(WHITE)
        shape.line.color.rgb = rgb(LINE)
        add_text(slide, cx + 0.10, y + 0.08, w - 0.2, 0.16, tag, 8, INK_MID, True, 'Aptos Mono', PP_ALIGN.CENTER)
        cx += w + 0.12


def title_slide(prs, s, i):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    decorate(slide, s, i)
    add_text(slide, 0.82, 1.45, 10.5, 1.85, s['title'], 34, INK, True)
    add_text(slide, 0.85, 3.55, 8.8, 1.2, s['body'], 16, INK_MID)
    add_tags(slide, s.get('tags', []), 0.85, 5.28)


def decorate(slide, s, i):
    bg = slide.background.fill
    bg.solid(); bg.fore_color.rgb = rgb(BG)
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(0.08), Inches(7.5))
    bar.fill.solid(); bar.fill.fore_color.rgb = rgb(ORANGE); bar.line.fill.background()
    add_kicker(slide, s['kicker'])
    add_footer(slide, s['footer'], i)


def generic_slide(prs, s, i):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    decorate(slide, s, i)
    add_text(slide, 0.82, 0.92, 10.8, 0.75, s['title'], 26, INK, True)
    layout = s['layout']
    if s.get('quote'):
        qbox = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.85), Inches(1.88), Inches(10.9), Inches(1.05))
        qbox.fill.background(); qbox.line.color.rgb = rgb(ORANGE); qbox.line.width = Pt(2)
        add_text(slide, 1.05, 1.95, 10.2, 0.54, s['quote'], 14, INK)
        add_text(slide, 1.05, 2.55, 10.1, 0.22, s.get('quote_src', ''), 8, INK_SOFT, True, 'Aptos Mono')
    if layout == 'quote_metrics':
        for col, (num, cap) in enumerate(s['metrics']):
            add_card(slide, 0.85 + col * 4.05, 3.25, 3.72, 1.55, cap, num=num)
    elif layout == 'metrics6':
        for idx, (num, cap) in enumerate(s['metrics']):
            add_card(slide, 0.85 + (idx % 3) * 4.05, 1.95 + (idx // 3) * 1.85, 3.72, 1.55, cap, num=num)
    elif layout == 'pipeline':
        for idx, (title, body) in enumerate(s['stages']):
            add_card(slide, 0.65 + idx * 2.48, 1.95, 2.24, 2.35, title, body)
        add_text(slide, 0.86, 4.75, 10.8, 0.6, s['body'], 13, INK_MID)
    elif layout == 'quote_cards':
        for idx, (title, body) in enumerate(s['cards']):
            add_card(slide, 0.85 + idx * 4.05, 3.35, 3.72, 1.65, title, body)
    elif layout == 'cards3_body':
        for idx, (title, body) in enumerate(s['cards']):
            add_card(slide, 0.85 + idx * 4.05, 2.05, 3.72, 1.75, title, body)
        add_text(slide, 0.86, 4.15, 10.8, 0.82, s['body'], 12, INK_MID)
    elif layout == 'cards4':
        for idx, (title, body) in enumerate(s['cards']):
            add_card(slide, 0.85 + (idx % 2) * 6.08, 1.95 + (idx // 2) * 1.85, 5.65, 1.55, title, body)
    elif layout == 'cards_metrics':
        for idx, (title, body) in enumerate(s['cards']):
            add_card(slide, 0.85 + idx * 4.05, 1.88, 3.72, 1.45, title, body)
        for idx, (num, cap) in enumerate(s['metrics']):
            add_card(slide, 0.85 + idx * 4.05, 3.72, 3.72, 1.45, cap, num=num)
    elif layout == 'checklist':
        all_items = [(x, False) for x in s['checks']] + [(x, True) for x in s['roadmap']]
        for idx, (item, road) in enumerate(all_items):
            col = idx % 2; row = idx // 2
            x = 0.85 + col * 5.95; y = 1.78 + row * 0.64
            shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(5.55), Inches(0.48))
            shape.fill.solid(); shape.fill.fore_color.rgb = rgb(WHITE)
            shape.line.color.rgb = rgb(LINE)
            add_text(slide, x + 0.12, y + 0.11, 0.25, 0.16, '>' if road else '+', 9, INK_SOFT if road else ORANGE, True, 'Aptos Mono')
            add_text(slide, x + 0.38, y + 0.09, 5.0, 0.20, item, 8, INK_MID)
    return slide


def write_pptx() -> None:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    # remove default first slide if present? python-pptx starts empty.
    for i, s in enumerate(slides, 1):
        if s['layout'] == 'title':
            title_slide(prs, s, i)
        else:
            generic_slide(prs, s, i)
    prs.save(PPTX_PATH)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_html()
    write_pptx()
    print(f'wrote {HTML_PATH.relative_to(OUT_DIR.parent.parent)}')
    print(f'wrote {PPTX_PATH.relative_to(OUT_DIR.parent.parent)}')
    print(f'wrote {len(slides)} individual slide HTML files in {SLIDES_DIR.relative_to(OUT_DIR.parent.parent)}')


if __name__ == '__main__':
    main()
