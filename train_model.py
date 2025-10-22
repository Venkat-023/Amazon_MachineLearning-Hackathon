"""
train_model.py
-----------------------------------
Train the SentenceTransformer + LightGBM model
for Amazon ML Hackathon product price prediction.
"""

# ================================
# 📦 IMPORTS
# ================================
import pandas as pd
import numpy as np
import re
import emoji
import nltk
import joblib
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")


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

def smape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_true - y_pred) / np.where(denom == 0, 1, denom)
    return np.mean(diff) * 100



df = pd.read_csv("structured_catalog_dataset.csv").fillna("")

# Normalize unit column
if "normalized_unit" in df.columns:
    df["normalized_unit"] = df["normalized_unit"].apply(normalize_unit)
elif "quantity_unit" in df.columns:
    df["normalized_unit"] = df["quantity_unit"].apply(normalize_unit)
else:
    df["normalized_unit"] = "unknown"

# Combine text fields
text_cols = [
    "item_name", "bullet_point_1", "bullet_point_2", "bullet_point_3",
    "bullet_point_4", "bullet_point_5", "product_description"
]
for col in text_cols:
    if col not in df.columns: df[col] = ""

print("🧹 Cleaning text columns...")
df["combined_text"] = df[text_cols].astype(str).agg(" ".join, axis=1)
df["combined_text"] = df["combined_text"].apply(clean_text)

target = "price"
cat_cols = ["brand","normalized_unit"]
num_cols = ["quantity_value","quantity_value_converted"]
text_col = "combined_text"

X_df = df.drop(columns=[target])
y = df[target]


print("🔡 Encoding categorical and numeric features...")
enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_cat = enc.fit_transform(X_df[cat_cols]).astype(np.float32)

scaler = StandardScaler()
X_num = scaler.fit_transform(X_df[num_cols].replace("",0).fillna(0)).astype(np.float32)


print("🔍 Loading SentenceTransformer (MiniLM)...")
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedder = SentenceTransformer(model_name, device="cuda")

print("⚙️ Generating text embeddings (GPU)...")
text_embeddings = embedder.encode(
    X_df[text_col].tolist(),
    batch_size=128,
    show_progress_bar=True,
    convert_to_numpy=True,
    device="cuda"
).astype(np.float32)

# Combine all features
X_all = np.hstack([text_embeddings, X_cat, X_num])
print(f"✅ Combined feature shape: {X_all.shape}")


X_train, X_valid, y_train, y_valid = train_test_split(X_all, y, test_size=0.15, random_state=42)

params = {
    "objective": "regression_l1",
    "boosting_type": "gbdt",
    "n_estimators": 800,
    "learning_rate": 0.05,
    "num_leaves": 256,
    "max_depth": 12,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "device": "gpu",
    "n_jobs": -1,
    "random_state": 42,
    "verbose": -1
}

print("🚀 Training LightGBM model...")
model = LGBMRegressor(**params)
model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric="l1",
    callbacks=[early_stopping(100), log_evaluation(0)]
)

y_pred = model.predict(X_valid)
smape_score = smape(y_valid, y_pred)
print(f"✅ Validation SMAPE: {smape_score:.4f}")


joblib.dump((model, embedder, enc, scaler, text_cols, cat_cols, num_cols), "final_transformer_lightgbm.pkl")
print("💾 Saved model and transformers → final_transformer_lightgbm.pkl")
