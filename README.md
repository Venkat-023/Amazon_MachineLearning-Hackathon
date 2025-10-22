Unstop ML Hackathon 2025 | Team AVIS | Product Price Prediction | Rank 1693 |Total Registered:82,790 | SMAPE 51.4

#  Amazon Machine Learning Hackathon 2025 — Product Price Prediction

This repository contains Team AVIS’s solution for the **Unstop ML Hackathon 2025**, where the task was to predict product prices from structured and unstructured catalog data.

🏆 Team Name: AVIS
📈 Final SMAPE: 51.4
🥇 Leaderboard Rank: #1693
⚙️ Frameworks: LightGBM + Sentence Transformers (MiniLM-L6-v2)
🧠 Hardware: GPU-accelerated training on Google Colab

🚀 Project Overview

The goal was to build a regression model that accurately predicts product prices using both structured features (brand, quantity, unit) and unstructured text (titles, bullet points, and product descriptions).

Component	Description
Text Encoder	SentenceTransformer – all-MiniLM-L6-v2
Model	LightGBM (GPU, regression_l1 objective)
Metric	SMAPE (Symmetric Mean Absolute Percentage Error)
Optimization	Early stopping, feature scaling, lemmatization, unit normalization
🧠 Model Architecture

Text Cleaning — remove emojis, punctuation, and stopwords

Embedding Generation — use SentenceTransformer (MiniLM-L6-v2)

Feature Fusion — combine embeddings + categorical + numeric features

Training — GPU-based LightGBM regressor

Evaluation — SMAPE metric

📊 Results
Metric	Score
Validation SMAPE	47.43
Public Leaderboard SMAPE	51.4
Final Rank	#1693 / 84,000
🧩 Tech Stack

🐍 Python 3.12

💡 LightGBM (GPU)

🤖 SentenceTransformers (MiniLM-L6-v2)

🧹 NLTK for text preprocessing

📦 scikit-learn, pandas, numpy, joblib

🧪 How to Run
# 1️⃣ Install dependencies
!pip install -q lightgbm sentence-transformers emoji nltk

# 2️⃣ Run the training script
python amazon_price_prediction.ipynb

# 3️⃣ Predict on new data
python inference_script.py --input test.csv --output predicted_prices.csv

🌟 Highlights

✅ Preprocessed 95K+ records combining structured and unstructured data
✅ Generated 384-dimensional text embeddings using MiniLM
✅ Optimized LightGBM with GPU acceleration
✅ Achieved SMAPE 51.4 — Top 2% out of 82,790 global participants
