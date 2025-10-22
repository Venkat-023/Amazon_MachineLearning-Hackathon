"""
predict_model.py
-----------------------------------
Load the trained model and generate
predicted prices for new test data.
"""

import pandas as pd
import numpy as np
import joblib
import re
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure stopwords available
nltk.download("stopwords")
nltk.download("wordnet")


def clean_text(text):
    if not isinstance(text, str): return ""
    text = emoji.replace_emoji(text, "")
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    words = text.split()
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(w) for w in words if w not in stop_words])

def normalize_unit(unit):
    if not isinstance(unit, str): return "unknown"
    u = unit.strip().lower()
    if u in ["ounce","ounces","oz","fl oz","fl ounce","fluid ounce","fl","fl. oz"]: return "oz"
    if u in ["pound","pounds","lb","lbs"]: return "lb"
    if u in ["gram","grams","g","milligram","mg"]: return "g"
    if u in ["kilogram","kilograms","kg"]: return "kg"
    if u in ["liter","litre","ltr","millilitre","milliliter","ml"]: return "l"
    if u in ["count","ct","pack","packet","bag","box","case","unit","piece","bottle","ea"]: return "count"
    if u in ["sq ft","foot","feet"]: return "sq_ft"
    if u in ["none","na","unknown","","n/a"]: return "unknown"
    return u


print("📦 Loading trained model and transformers...")
model, embedder, enc, scaler, text_cols, cat_cols, num_cols = joblib.load("final_transformer_lightgbm.pkl")


def predict_new(df_test_path, output_path="predicted_prices_transformer.csv"):
    print(f"📘 Reading test data: {df_test_path}")
    df_test = pd.read_csv(df_test_path).fillna("")

    # Normalize unit column
    if "normalized_unit" in df_test.columns:
        df_test["normalized_unit"] = df_test["normalized_unit"].apply(normalize_unit)
    elif "quantity_unit" in df_test.columns:
        df_test["normalized_unit"] = df_test["quantity_unit"].apply(normalize_unit)
    else:
        df_test["normalized_unit"] = "unknown"
    for col in text_cols:
        if col not in df_test.columns: df_test[col] = ""
    df_test["combined_text"] = df_test[text_cols].astype(str).agg(" ".join, axis=1)
    df_test["combined_text"] = df_test["combined_text"].apply(clean_text)

    X_cat_test = enc.transform(df_test[cat_cols]).astype(np.float32)
    X_num_test = scaler.transform(df_test[num_cols].replace("",0).fillna(0)).astype(np.float32)

    print("⚙️ Generating text embeddings (GPU)...")
    text_emb_test = embedder.encode(
        df_test["combined_text"].tolist(),
        batch_size=128,
        show_progress_bar=True,
        convert_to_numpy=True,
        device="cuda"
    ).astype(np.float32)

    X_test_all = np.hstack([text_emb_test, X_cat_test, X_num_test])
    preds = model.predict(X_test_all)

    if "sample_id" not in df_test.columns:
        df_test["sample_id"] = np.arange(len(df_test))

    submission = pd.DataFrame({"sample_id": df_test["sample_id"], "price": preds})
    submission.to_csv(output_path, index=False)
    print(f"✅ Predictions saved → {output_path}")
    print(submission.head())


if __name__ == "__main__":
    predict_new("test_dataset.csv")
