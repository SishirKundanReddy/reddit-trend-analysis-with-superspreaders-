import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

CHART_DIR = "output/charts/"


def ensure_chart_dir():
    if not os.path.exists(CHART_DIR):
        os.makedirs(CHART_DIR)


def bar_chart(series, title, xlabel, filename, top_n=15):
    ensure_chart_dir()
    plt.figure(figsize=(10, 6))
    series.head(top_n).plot(kind="barh")
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(CHART_DIR + filename)
    plt.close()


def sentiment_distribution(df):
    ensure_chart_dir()
    plt.figure(figsize=(8, 5))
    sns.histplot(df["avg_compound"], bins=20, kde=True)
    plt.title("Distribution of Comment Sentiment (Compound Score)")
    plt.xlabel("Sentiment Score")
    plt.tight_layout()
    plt.savefig(CHART_DIR + "sentiment_distribution.png")
    plt.close()


def engagement_by_label(df):
    ensure_chart_dir()
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="engagement_label", y="engagement_score", data=df)
    plt.title("Engagement Score by Trend Category")
    plt.tight_layout()
    plt.savefig(CHART_DIR + "engagement_by_label.png")
    plt.close()


def trend_keyword_frequency(df):
    ensure_chart_dir()

    # explode the keyword list
    keywords = df["trend_keywords"].dropna().str.split(", ")
    keyword_series = keywords.explode().value_counts()

    bar_chart(
        keyword_series,
        "Most Common Trend Keywords",
        "Frequency",
        "trend_keyword_frequency.png"
    )


def subreddit_scores(df):
    ensure_chart_dir()
    avg_scores = df.groupby("subreddit")["score"].mean().sort_values(ascending=False)

    bar_chart(
        avg_scores,
        "Average Score by Subreddit",
        "Average Score",
        "subreddit_avg_scores.png"
    )


def correlation_heatmap(df):
    ensure_chart_dir()
    numeric_df = df[["score", "num_comments", "engagement_score",
                     "avg_neg", "avg_neu", "avg_pos", "avg_compound"]]

    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(CHART_DIR + "correlation_heatmap.png")
    plt.close()


def trending_songs(df):
    ensure_chart_dir()
    song_counts = df["song"].value_counts()

    bar_chart(
        song_counts,
        "Most Frequently Mentioned Songs",
        "Mentions",
        "top_songs.png"
    )


def trending_artists(df):
    ensure_chart_dir()
    artist_counts = df["artist"].value_counts()

    bar_chart(
        artist_counts,
        "Most Frequently Mentioned Artists",
        "Mentions",
        "top_artists.png"
    )


def scatter_engagement(df):
    ensure_chart_dir()
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=df["score"],
        y=df["num_comments"],
        hue=df["engagement_label"]
    )
    plt.xlabel("Score (Upvotes)")
    plt.ylabel("Number of Comments")
    plt.title("Score vs Comments Colored by Trend Category")
    plt.tight_layout()
    plt.savefig(CHART_DIR + "scatter_score_comments.png")
    plt.close()

def generate_all_visualizations():
    print("Loading dataset...")
    trend_df = pd.read_csv("output/trend_dataset.csv")
    sent_df = pd.read_csv("output/reddit_comment_sentiment.csv")

    # ----------------------------
    # Ensure a common join key exists
    # ----------------------------
    if "permalink" not in trend_df.columns:
        if "url" in trend_df.columns:
            trend_df["permalink"] = trend_df["url"]
        else:
            # try other common candidates
            for candidate in ("link", "post_url", "permalink_url"):
                if candidate in trend_df.columns:
                    trend_df["permalink"] = trend_df[candidate]
                    break

    if "permalink" not in sent_df.columns:
        if "url" in sent_df.columns:
            sent_df["permalink"] = sent_df["url"]

    # If still missing, warn and try to merge on title as a last resort
    if "permalink" not in trend_df.columns or "permalink" not in sent_df.columns:
        print("[WARN] permalink missing in one of the datasets. Falling back to merge on 'title' if possible.")
        if "title" in trend_df.columns and "title" in sent_df.columns:
            df = pd.merge(trend_df, sent_df, on="title", how="left")
        else:
            # final fallback: join by index (not recommended)
            print("[WARN] No reliable join key found. Merging by index (may misalign rows).")
            trend_df = trend_df.reset_index(drop=True)
            sent_df = sent_df.reset_index(drop=True)
            # pad shorter DF if needed
            n = max(len(trend_df), len(sent_df))
            trend_df = trend_df.reindex(range(n))
            sent_df = sent_df.reindex(range(n))
            df = pd.concat([trend_df, sent_df.add_suffix("_sent")], axis=1)
    else:
        # Normal case: merge on permalink
        df = pd.merge(trend_df, sent_df, on="permalink", how="left")

    # ----------------------------
    # Normalize important columns so visualizations don't crash
    # ----------------------------
    # Ensure 'song' exists
    if "song" not in df.columns:
        for candidate in ("song_x", "song_y", "song_sent"):
            if candidate in df.columns:
                df["song"] = df[candidate]
                break
        else:
            # fallback to title as a safer alternative
            df["song"] = df.get("title", "").fillna("unknown")

    # Ensure 'artist' exists
    if "artist" not in df.columns:
        for candidate in ("artist_x", "artist_y", "artist_sent"):
            if candidate in df.columns:
                df["artist"] = df[candidate]
                break
        else:
            df["artist"] = df.get("artist", "").fillna("")

    # Ensure numeric columns exist and are numeric (avoid type errors)
    for col in ["score", "num_comments", "engagement_score", "avg_compound", "avg_pos", "avg_neg", "avg_neu"]:
        if col not in df.columns:
            df[col] = 0
        # try converting to numeric safely
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # For trend_keywords, ensure it's a string column (no crash when splitting)
    if "trend_keywords" not in df.columns:
        df["trend_keywords"] = ""

    print("Generating visualizations...")

    trending_songs(df)
    trending_artists(df)
    subreddit_scores(df)
    sentiment_distribution(df)
    engagement_by_label(df)
    trend_keyword_frequency(df)
    correlation_heatmap(df)
    scatter_engagement(df)

    print("Charts saved in output/charts/")

