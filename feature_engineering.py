# feature_engineering.py

# emotion_words = [
#     "shocking", "breaking", "unbelievable",
#     "amazing", "terrifying", "secret", "exposed"
# ]

# import numpy as np
import math

LABEL_DETAILS = {
    "Normal News": {
        "seo_friendly": True,
        "google_search_friendly": True,
        "social_media_friendly": False,
        "where_it_works": [
            "Google Search",
            "Google News",
            "Professional News Websites"
        ],
        "meaning": "Professional, factual, low-emotion headline",
        "suggestion": (
            "No Suggestion Needed Headline is Perfect"
        )
    },

    "Medium Sensational": {
        "seo_friendly": True,
        "google_search_friendly": True,
        "social_media_friendly": True,
        "where_it_works": [
            "Google Search",
            "Google Discover",
            "Facebook",
            "Twitter (X)"
        ],
        "meaning": "Balanced headline with emotion and information",
        "suggestion": (
            "This headline is well-balanced. "
            "You may slightly enhance emotional wording, "
            "but keep it factual to maintain SEO quality."
        )
    },

    "High Clickbait": {
        "seo_friendly": False,
        "google_search_friendly": False,
        "social_media_friendly": True,
        "where_it_works": [
            "YouTube Titles",
            "Facebook Pages",
            "TikTok",
            "Instagram Reels"
        ],
        "meaning": "Highly emotional or exaggerated headline",
        "suggestion": (
            "To make this headline SEO-friendly, reduce emotional exaggeration, "
            "shorten the length, and include clear factual keywords."
        )
    }
}

CLUSTER_CENTROIDS = {
    0: {  # Normal News
        "headline_word_count": 8.040129,
        "headline_char_count": 53.080995,
        "has_question": 0.0,
        "capital_ratio": 0.102751,
        "has_number": 0.107583,
        "emotion_score": 0.048066
    },
    1: {  # Medium Sensational
        "headline_word_count": 11.490042,
        "headline_char_count": 67.728429,
        "has_question": 1.0,
        "capital_ratio": 0.104237,
        "has_number": 0.184600,
        "emotion_score": 0.130362
    },
    2: {  # High Clickbait
        "headline_word_count": 14.780495,
        "headline_char_count": 91.142799,
        "has_question": 0.000425,
        "capital_ratio": 0.099132,
        "has_number": 0.594077,
        "emotion_score": 0.285605
    }
}

cluster_labels = {
    0: "Normal News",
    1: "Medium Sensational",
    2: "High Clickbait"
}


emotion_words = [

    # 🔥 Shock / Surprise
    "shocking", "shock", "shocked", "unbelievable", "unreal",
    "jaw-dropping", "mind-blowing", "mind blowing",
    "insane", "crazy", "unexpected", "unimaginable",
    "can’t believe", "hard to believe", "never expected",

    # 🚨 Breaking / Urgency
    "breaking", "breaking news", "urgent", "just in",
    "alert", "exclusive", "leaked", "leak",
    "developing", "confirmed", "now", "right now",
    "live", "happening now", "latest update",

    # 😱 Fear / Danger
    "terrifying", "horrifying", "dangerous", "deadly",
    "threat", "panic", "warning", "emergency",
    "disaster", "catastrophic", "crisis", "chaos",
    "fear spreads", "people terrified", "risk alert",

    # 🕵️ Curiosity / Mystery
    "secret", "secrets", "revealed", "exposed", "hidden",
    "unknown", "truth", "truth behind",
    "what happened", "what really happened",
    "you won’t believe", "this is why",
    "mystery", "shocking truth", "real reason",

    # 📢 Exaggeration / Hype
    "amazing", "incredible", "epic", "legendary",
    "massive", "huge", "biggest", "best", "worst",
    "never seen", "once in a lifetime",
    "record breaking", "history made", "unprecedented",

    # ❤️ Emotion (Positive / Negative)
    "heartbreaking", "emotional", "devastating",
    "outrage", "furious", "angry",
    "sad", "horrible", "tragic", "painful",
    "happy", "excited", "emotional moment",
    "tears", "crying", "grieving",

    # 🧲 Clickbait Phrases
    "top", "top 10", "reasons", "things", "facts",
    "ways", "tricks", "what happens next",
    "this changed everything",
    "will shock you", "must watch",
    "people are shocked", "internet reacts",
    "goes viral", "everyone is talking about this",

    # 🔠 Capitalized Clickbait
    "SHOCKING", "BREAKING", "URGENT",
    "UNBELIEVABLE", "EXCLUSIVE",
    "THIS WILL SHOCK YOU",
]






# def emotion_word_count(text):
#     text = text.lower()
#     return sum(1 for w in emotion_words if w in text)

def feature_contribution_percent(input_features, cluster_id):
    centroid = CLUSTER_CENTROIDS[cluster_id]
    diffs = {}

    total_diff = 0
    for k in centroid:
        diff = abs(input_features[k] - centroid[k])
        diffs[k] = diff
        total_diff += diff

    percentages = {}
    for k, v in diffs.items():
        percentages[k] = round((v / total_diff) * 100, 2) if total_diff != 0 else 0

    return percentages


def calculate_emotion_score(text):
    text = text.lower()
    score = 0
    for w in emotion_words:
        if w in text:
            score += 2 if len(w) > 6 else 1
    # Dynamic scaling without being tiny
    scaled = score / max(len(text.split()) * 2, 1)  # normalize by words * max weight
    # Use tanh to compress high scores but keep range meaningful
    return math.tanh(scaled * 10)  # multiplier adjust karo

# def calculate_emotion_score(text):
#     text = text.lower()
#     score = 0
#     for w in emotion_words:
#         if w in text:
#             score += 2 if len(w) > 6 else 1
    
#     # ✅ dynamic percentage
#     # assume max possible score = len(text.split()) * 2  (agar har word match ho jaye max weight)
#     max_score = len(text.split()) * 2
#     if max_score == 0:
#         return 0.0
#     return score / max_score  # returns 0-1


def extract_features(headline: str):
    # 1. word count
    headline_word_count = len(headline.split())

    # 2. character count
    headline_char_count = len(headline)

    # 3. question mark
    has_question = 1 if "?" in headline else 0

    # 4. capital ratio
    capital_ratio = sum(1 for c in headline if c.isupper()) / max(len(headline), 1)

    # 5. number presence
    has_number = 1 if any(char.isdigit() for char in headline) else 0

    # 6. emotion score (tumhara existing function)
    emotion_score = calculate_emotion_score(headline)

    return {
        "headline_word_count": headline_word_count,
        "headline_char_count": headline_char_count,
        "has_question": has_question,
        "capital_ratio": capital_ratio,
        "has_number": has_number,
        "emotion_score": emotion_score
    }

