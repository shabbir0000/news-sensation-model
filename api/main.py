from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

from api.feature_engineering import LABEL_DETAILS, extract_features, feature_contribution_percent

app = FastAPI()
# -----------------------
# Load saved models
# -----------------------
# scaler = joblib.load("models/sensational_scaler_model_v1.1.pkl")
# kmeans = joblib.load("models/sensational_kmeans_model_v1.1.pkl")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

scaler = joblib.load(
    os.path.join(BASE_DIR, "models", "sensational_scaler_model_v1.1.pkl")
)

kmeans = joblib.load(
    os.path.join(BASE_DIR, "models", "sensational_kmeans_model_v1.1.pkl")
)
# -----------------------
# Cluster labels
# -----------------------
cluster_labels = {
    0: "Normal News",
    1: "Medium Sensational",
    2: "High Clickbait"
}

# -----------------------
# FastAPI app
# -----------------------


# -----------------------
# Request schema
# -----------------------
class NewsRequest(BaseModel):
    headline: str

# -----------------------
# Root endpoint
# -----------------------
@app.get("/")
def home():
    return {"message": "News Headline Prediction API is running"}

# -----------------------
# Prediction endpoint
# -----------------------
@app.post("/predict")
def predict_news(request: NewsRequest):

    headline = request.headline

    # Step 1: Feature extraction
    extracted = extract_features(headline)

    feature_vector = [
        extracted["headline_word_count"],
        extracted["headline_char_count"],
        extracted["has_question"],
        extracted["capital_ratio"],
        extracted["has_number"],
        extracted["emotion_score"]
    ]

    # Step 2: Scaling (NO fit)
    features_scaled = scaler.transform([feature_vector])

    # Step 3: Prediction
    cluster = int(kmeans.predict(features_scaled)[0])
    label = cluster_labels.get(cluster, "Unknown")

    # Step 4: Explainability
    contributions = feature_contribution_percent(extracted, cluster)

    top_features = sorted(
        contributions.items(),
        key=lambda x: x[1],
        reverse=True
    )[:2]

    reason = f"Prediction based mainly on {top_features[0][0]} and {top_features[1][0]}"

    # Step 5: Label details
    label_info = LABEL_DETAILS.get(label, {})

    return {
        "headline": headline,
        "prediction": label,
        "cluster_id": cluster,

        "feature_values": extracted,
        "feature_contribution_percent": contributions,

        "reason": reason,

        # 🔥 NEW INTELLIGENT PART
        "headline_analysis": {
            "meaning": label_info.get("meaning"),
            "seo_friendly": label_info.get("seo_friendly"),
            "google_search_friendly": label_info.get("google_search_friendly"),
            "social_media_friendly": label_info.get("social_media_friendly"),
            "best_platforms": label_info.get("where_it_works"),
            "improvement_suggestion": label_info.get("suggestion")
        }
    }
