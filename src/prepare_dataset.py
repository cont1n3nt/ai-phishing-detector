import pandas as pd
from features import clean_text

INPUT_FILE = "../data/raw/dataset_final.csv"
OUTPUT_FILE = "../data/preprocess/cleaned_dataset.csv"


def main():
    df = pd.read_csv(INPUT_FILE)
    
    print(f"[INFO] Исходный датасет: {df.shape[0]} строк")
    df.drop_duplicates(subset=['text'], inplace=True)
    
    print(f"[INFO] После удаления дубликатов: {df.shape[0]} строк")
    df['clean_text'] = df['text'].apply(clean_text)
    
    df = df[df['clean_text'].str.len() > 5]
    print(f"[INFO] После удаления коротких/пустых строк: {df.shape[0]} строк")

    print("[INFO] Баланс классов:")
    print(df['label'].value_counts())

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"[INFO] Очищенный датасет сохранён: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()