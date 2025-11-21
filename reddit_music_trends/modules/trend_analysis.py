import pandas as pd

# Trend-related keywords to detect in the text
TREND_KEYWORDS = [
    "trend", "trending", "viral", "tiktok", "blowing up", "blow up",
    "charting", "billboard", "hit", "popular", "streaming boost",
    "going viral", "exploding", "hot", "number one", "rising",
    "breakout", "new hit", "buzz", "hype", "hyped", "fresh"
]


def detect_keywords(text):
    """Detect any trend-related keywords."""
    hits = []
    if not isinstance(text, str):
        return ""

    t = text.lower()
    for k in TREND_KEYWORDS:
        if k in t:
            hits.append(k)

    return ", ".join(hits)


def engagement_level(score, comments):
    """Categorize engagement heuristically."""
    if score > 500 or comments > 200:
        return "TRENDING"
    if score > 200 or comments > 80:
        return "EMERGING"
    if score > 50:
        return "STABLE"
    return "LOW"


def compute_engagement_score(score, comments):
    """
    A combined score (used later in ML):
    Weighted sum: 70% score + 30% comments.
    """
    return (0.7 * score) + (0.3 * comments)


def analyze_trends(df):
    """Add trend metrics, engagement labels, and save dataset."""
    df = df.copy()

    # Detect trend keywords
    df["trend_keywords"] = df["clean_text"].apply(detect_keywords)

    # Engagement labels
    df["engagement_label"] = df.apply(
        lambda r: engagement_level(r["score"], r["num_comments"]),
        axis=1
    )

    # Engagement numeric score
    df["engagement_score"] = df.apply(
        lambda r: compute_engagement_score(r["score"], r["num_comments"]),
        axis=1
    )

    # Filter relevant posts
    df_trend = df[
        (df["song"].notnull()) |
        (df["trend_keywords"] != "")
    ].copy()

    df_trend.to_csv("output/trend_dataset.csv", index=False)
    print("Saved trend dataset: output/trend_dataset.csv")

    return df_trend
