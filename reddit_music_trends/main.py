import os

# Import your modules
from modules.reddit_fetch import fetch_all
from modules.data_cleaning import clean_dataset
from modules.trend_analysis import analyze_trends
from modules.sentiment_analysis import process_sentiment
from modules.ml_model import train_models
from modules.visualization import generate_all_visualizations
from modules.superspreaders import detect_superspreaders

OUTPUT_DIR = "output/"
CHART_DIR = "output/charts/"


def ensure_folders():
    """Create output folders if they don't exist."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    if not os.path.exists(CHART_DIR):
        os.makedirs(CHART_DIR)


def main():
    print("\n=== Reddit Music Trends Research Pipeline ===\n")

    ensure_folders()

    # 1. Fetch Reddit posts
    print("\n[1] Fetching Reddit posts...")
    raw_df = fetch_all()

    # Save raw posts for superspreaders
    raw_df.to_csv("output/raw_posts.csv", index=False)

    # 2. Clean text + extract songs/artists
    print("\n[2] Cleaning dataset...")
    cleaned_df = clean_dataset(raw_df)

    # 3. Detect trend keywords + categorize engagement
    print("\n[3] Analyzing trends...")
    trend_df = analyze_trends(cleaned_df)

    # 4. Comment scraping + sentiment analysis
    print("\n[4] Scraping comments + performing sentiment analysis...")
    sentiment_df = process_sentiment(trend_df)

    # ➜ This step creates output/comments_raw.csv inside process_sentiment()

    # 5. Detect superspreaders (NOW comments_raw.csv exists)
    print("\n[5] Detecting superspreaders...")
    detect_superspreaders(
        posts_csv="output/raw_posts.csv",
        comments_csv="output/comments_raw.csv",
        out_csv="output/superspreaders.csv",
        out_graph="output/charts/superspreaders_graph.png",
        top_n=50
    )

    # 6. Build ML model for “Trending vs Non-trending”
    print("\n[6] Training machine learning model...")
    train_models()

    # 7. Generate all visualizations
    print("\n[7] Generating charts...")
    generate_all_visualizations()

    print("\n\n=== Pipeline Complete! 🚀 ===")
    print("All outputs saved in /output/")
    print("Charts saved in /output/charts/")
    print("Model saved as output/model.pkl\n")


if __name__ == "__main__":
    main()
