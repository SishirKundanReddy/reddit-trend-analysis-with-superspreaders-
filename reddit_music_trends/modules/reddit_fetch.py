import praw
import pandas as pd

# -------------------------------------------
# ENTER YOUR REAL CREDENTIALS HERE
# -------------------------------------------
CLIENT_ID = "_oOe-QiYU5GuEnLeZlDccQ"
CLIENT_SECRET = "u_8pWkk2UbPIaXUWxNJ1nygAotVZ1g"
USERNAME = "Emotional_Kiwi_5256"
PASSWORD = "Cherry@1"
USER_AGENT = "music_trends_app:v1.0 (by u/YOUR_USERNAME)"

SUBREDDITS = ["Music", "PopHeads", "HipHopHeads"]
POST_LIMIT = 200


def fetch_all():
    """Fetch posts using official Reddit API (PRAW)."""

    reddit = praw.Reddit(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        user_agent=USER_AGENT,
        username=USERNAME,
        password=PASSWORD
    )

    print("\nAuthenticated as:", reddit.user.me())

    all_posts = []

    for sub in SUBREDDITS:
        print(f"Fetching from r/{sub}...")

        for post in reddit.subreddit(sub).hot(limit=POST_LIMIT):
            all_posts.append({
                "subreddit": sub,
                "title": post.title,
                "score": post.score,
                "num_comments": post.num_comments,
                "created_utc": post.created_utc,
                "url": post.url
            })

    df = pd.DataFrame(all_posts)
    df.to_csv("output/raw_posts.csv", index=False)

    print(f"\nSaved {len(df)} posts → output/raw_posts.csv")
    return df
