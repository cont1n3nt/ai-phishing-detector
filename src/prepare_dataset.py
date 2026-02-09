import pandas as pd
import re

INPUT_FILE = "../data/raw/dataset_final.csv"
OUTPUT_FILE = "../data/preprocess/cleaned_dataset.csv"

def clean_text(text: str) -> str:
    """Очистка текста: lowercase, удаление ссылок и спецсимволов"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)          # удалить URL
    text = re.sub(r"[^a-zA-Z\s]", "", text)     # оставить только буквы
    text = re.sub(r"\s+", " ", text).strip()    # удалить лишние пробелы
    return text

# ---------- Основной код ----------
def main():
    # 1️⃣ Загружаем датасет
    df = pd.read_csv(INPUT_FILE)
    print(f"[INFO] Исходный датасет: {df.shape[0]} строк")

    # 2️⃣ Удаляем дубликаты по тексту
    df.drop_duplicates(subset=['text'], inplace=True)
    print(f"[INFO] После удаления дубликатов: {df.shape[0]} строк")

    # 3️⃣ Чистим текст
    df['clean_text'] = df['text'].apply(clean_text)

    # 4️⃣ Удаляем пустые и слишком короткие тексты
    df = df[df['clean_text'].str.len() > 5]
    print(f"[INFO] После удаления коротких/пустых строк: {df.shape[0]} строк")

    # 5️⃣ Проверяем баланс классов
    print("[INFO] Баланс классов:")
    print(df['label'].value_counts())

    # 6️⃣ Сохраняем очищенный датасет
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"[INFO] Очищенный датасет сохранён: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()