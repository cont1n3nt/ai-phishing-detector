import pandas as pd
import re

df_base = pd.read_csv("../data/phishing_email.csv")
df_1 = pd.read_csv("../data/Enron.csv")
df_2 = pd.read_csv("../data/Ling.csv")

def normalize_subject_body(df):
    df = df[["subject", "body", "label"]].copy()
    df["text"] = df["subject"].fillna("") + " " + df["body"].fillna("")
    df = df[["text", "label"]]
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    return df

df_1 = normalize_subject_body(df_1)
df_2 = normalize_subject_body(df_2)

df_base = df_base.rename(columns={"text_combined": "text"})
df_base = df_base[["text","label"]]
df_base["text"] = df_base["text"].astype(str)
df_base["label"] = df_base["label"].astype(int)

df_all = pd.concat([df_base, df_1, df_2], ignore_index=True)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|[\w\.-]+@[\w\.-]+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = " ".join([w for w in text.split() if len(w) > 2])
    text = re.sub(r"\s+", " ", text).strip()
    return text

df_all["text"] = df_all["text"].fillna("").astype(str)

df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)
df_all.to_csv("../data/dataset_final.csv", index=False)

print("Final dataset saved:", df_all.shape)
