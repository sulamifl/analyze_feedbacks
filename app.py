import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

MODEL_PATH = "final_model" if os.path.isdir("final_model") else "edelweiss1/rubert-tiny2-ru-reviews-sentiment"

LABELS = {0: "Отрицательный", 1: "Средний / смешанный", 2: "Положительный"}
MAX_ROWS = 5000

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()


def predict_probs(texts, batch_size=32):
    """Принимает список текстов, возвращает вероятности: по 3 числа на каждый отзыв."""
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, truncation=True, max_length=128, padding=True, return_tensors="pt")
        with torch.no_grad():
            logits = model(**enc).logits
        all_probs.extend(torch.softmax(logits, dim=-1).tolist())
    return all_probs

def predict_one(text):
    if not text or not text.strip():
        return {}
    probs = predict_probs([text])[0]
    return {LABELS[i]: probs[i] for i in range(3)}


def analyze_csv(file, column):
    if file is None:
        raise gr.Error("Сначала загрузите CSV-файл.")

    path = file if isinstance(file, str) else file.name
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="cp1251")

    column = column.strip()
    if column not in df.columns:
        raise gr.Error(f"Колонки «{column}» нет в файле. Есть такие: {', '.join(map(str, df.columns))}")

    df = df.dropna(subset=[column]).head(MAX_ROWS).copy()
    if df.empty:
        raise gr.Error("В выбранной колонке нет текстов.")
    
    probs = predict_probs(df[column].astype(str).tolist())
    df["Тональность"] = [LABELS[p.index(max(p))] for p in probs]
    df["Уверенность"] = [round(max(p), 2) for p in probs]

    counts = df["Тональность"].value_counts().reindex(list(LABELS.values()), fill_value=0)
    summary = pd.DataFrame({
        "Тональность": counts.index,
        "Отзывов": counts.values,
        "Доля, %": (counts.values / len(df) * 100).round(1),
    })

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(summary["Тональность"], summary["Отзывов"], color=["#d9534f", "#999999", "#5cb85c"])
    ax.set_ylabel("Количество тозывов")
    ax.set_title(f"Тональность отзывов (всего: {len(df)})")
    plt.tight_layout()

    out_path = "results.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    return summary, fig, df[[column, "Тональность", "Уверенность"]].head(20), out_path


single_tab = gr.Interface(
    fn=predict_one,
    inputs=gr.Textbox(lines=4, label="Текст отзыва", placeholder="Вставьте отзыв покупателя..."),
    outputs=gr.Label(num_top_classes=3, label="Тональность"),
    description="Модель определеяет, отрицательный, средний или положительный отзыв. "
                "Обучена на отзывах о женской одежде.",
    examples=[
        ["Платье красивое, но села после стирки"],
        ["Размер не подошёл, возвращаю"],
        ["Отличное качество, рекомендую продавца"],
    ],
)

csv_tab = gr.Interface(
    fn=analyze_csv,
    inputs=[
        gr.File(label="CSV-файл с отзывами", file_types=[".csv"]),
        gr.Textbox(value="text", label="Название колонки с текстом отзыва"),
    ],
    outputs = [
        gr.Dataframe(label="Сводка"),
        gr.Plot(label="Распространение"),
        gr.Dataframe(label="Первые 20 отзывов с ответами модели"),
        gr.File(label="Скачать полный результат"),
    ],
    description=f"Загрузите CSV, и модель разберет все отзывы (до {MAX_ROWS} строк).",
)

demo = gr.TabbedInterface(
    [single_tab, csv_tab],
    ["Один отзыв", "Файл CSV"],
    title="Тональность русскоязычных отзывов",
)

if __name__ == "__main__":
    demo.launch()