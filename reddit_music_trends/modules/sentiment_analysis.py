# modules/sentiment_analysis.py
import requests
import pandas as pd
import time
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

USER_AGENT = "music_trend_research:v1.0 (by u/yourusername)"

# Helper: ensure output folder exists
os.makedirs("output", exist_ok=True)

def _full_permalink(permalink):
    """Return a full reddit URL for a permalink or url-like string."""
    if not isinstance(permalink, str):
        return ""
    p = permalink.strip()
    if p.startswith("http"):
        return p
    if p.startswith("/"):
        return "https://www.reddit.com" + p
    # if it's just an id (t3_xxx or id), try leaving as-is
    return p

def fetch_comments(permalink, limit=50, depth=1):
    """
    Fetch comments for a post permalink (handles nested replies to limited depth).
    Returns a list of dicts: {"author": ..., "body": ...}
    """
    url = _full_permalink(permalink)
    if not url:
        return []
    # ensure .json endpoint
    if not url.endswith(".json"):
        if url.endswith("/"):
            url = url[:-1]
        url = url + ".json?limit=" + str(limit)

    headers = {"User-Agent": USER_AGENT}
    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code != 200:
            return []
        data = r.json()
    except Exception:
        return []

    comments = []

    def walk_comment_tree(items):
        """Recursively walk a comment listing and append (author, body)."""
        if not isinstance(items, list):
            return
        for it in items:
            if not isinstance(it, dict):
                continue
            kind = it.get("kind")
            data = it.get("data", {})
            if kind == "t1":  # comment
                author = data.get("author") or "[deleted]"
                body = data.get("body") or ""
                comments.append({"author": author, "body": body})
                # nested replies
                replies = data.get("replies")
                if isinstance(replies, dict):
                    children = replies.get("data", {}).get("children", [])
                    walk_comment_tree(children)
            # sometimes more nested listings appear; ignore other kinds here

    # comments listing is typically at index 1
    if isinstance(data, list) and len(data) > 1:
        comment_listing = data[1].get("data", {}).get("children", [])
        walk_comment_tree(comment_listing)

    return comments

def analyze_sentiment(text):
    """Return VADER polarity scores for given text."""
    if not isinstance(text, str) or text.strip() == "":
        return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}
    return analyzer.polarity_scores(text)

def process_sentiment(df, comment_limit_per_post=50):
    """
    Main entrypoint:
    - df: DataFrame of posts (must contain 'permalink' or 'url' or similar).
    - returns a DataFrame with averaged sentiment per post and also writes raw comments file.
    """
    print("Fetching comments & computing sentiment...")

    # prepare lists to collect
    comments_summary = []
    raw_comments = []

    # choose permalink column
    if "permalink" in df.columns:
        pl_col = "permalink"
    elif "url" in df.columns:
        pl_col = "url"
    else:
        # try to find any column that looks like a link/id
        pl_col = None
        for c in df.columns:
            if "link" in c.lower() or "post" in c.lower() or "id" in c.lower():
                pl_col = c
                break
        if pl_col is None:
            # fallback to title as identifier
            pl_col = df.columns[0]

    for i, row in df.iterrows():
        permalink = row.get(pl_col, "")
        title = row.get("title", "")
        song = row.get("song", "")
        artist = row.get("artist", "")

        comments = fetch_comments(permalink, limit=comment_limit_per_post)
        # collect raw authors for superspreader analysis
        for c in comments:
            raw_comments.append({
                "post_id": _full_permalink(permalink) or str(i),
                "comment_author": c.get("author", "[deleted]")
            })

        # compute sentiment per comment
        sentiment_scores = [analyze_sentiment(c.get("body", "")) for c in comments]

        if sentiment_scores:
            avg_neg = sum(s["neg"] for s in sentiment_scores) / len(sentiment_scores)
            avg_neu = sum(s["neu"] for s in sentiment_scores) / len(sentiment_scores)
            avg_pos = sum(s["pos"] for s in sentiment_scores) / len(sentiment_scores)
            avg_comp = sum(s["compound"] for s in sentiment_scores) / len(sentiment_scores)
        else:
            avg_neg = avg_neu = avg_pos = avg_comp = 0.0

        comments_summary.append({
            "permalink": _full_permalink(permalink) or "",
            "title": title,
            "song": song,
            "artist": artist,
            "avg_neg": avg_neg,
            "avg_neu": avg_neu,
            "avg_pos": avg_pos,
            "avg_compound": avg_comp
        })

        # polite sleep to avoid hammering Reddit
        time.sleep(0.5)

    # Save raw comments for superspreader analysis (deduplicated)
    try:
        if raw_comments:
            raw_df = pd.DataFrame(raw_comments)
            # normalize column names
            if "post_id" in raw_df.columns and "comment_author" in raw_df.columns:
                raw_df.to_csv("output/comments_raw.csv", index=False)
                print("Saved raw comments: output/comments_raw.csv")
            else:
                # fallback: save whatever we have
                raw_df.to_csv("output/comments_raw.csv", index=False)
                print("Saved raw comments (fallback format): output/comments_raw.csv")
        else:
            # create an empty file if no comments were found
            pd.DataFrame(columns=["post_id", "comment_author"]).to_csv("output/comments_raw.csv", index=False)
            print("Saved empty raw comments: output/comments_raw.csv")
    except Exception as e:
        print("Warning: could not save raw comments:", e)

    # Build summary DataFrame
    df_sent = pd.DataFrame(comments_summary)

    # Label final sentiment
    if not df_sent.empty:
        df_sent["sentiment_label"] = df_sent["avg_compound"].apply(
            lambda c: "positive" if c > 0.1 else ("negative" if c < -0.1 else "neutral")
        )
    else:
        df_sent = pd.DataFrame(columns=["permalink", "title", "song", "artist",
                                        "avg_neg", "avg_neu", "avg_pos", "avg_compound", "sentiment_label"])

    # Save sentiment summary
    try:
        df_sent.to_csv("output/reddit_comment_sentiment.csv", index=False)
        print("Saved comment sentiment: output/reddit_comment_sentiment.csv")
    except Exception as e:
        print("Warning: could not save reddit_comment_sentiment.csv:", e)

    return df_sent
