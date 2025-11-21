import re
import pandas as pd

# Regex patterns
SONG_PATTERN = r"\"([^\"]+)\""                 # text inside quotes: "Song Name"
ARTIST_PATTERN = r"(?i)(?:by|from)\s+([A-Za-z0-9 .'-]+)"  # captures the artist name
CLEAN_HTML = re.compile("<.*?>")               # remove HTML tags


def clean_text(text):
    """Clean unwanted artifacts from text."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(CLEAN_HTML, "", text)        # remove HTML
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)           # collapse whitespace

    return text.strip()


def extract_song(text):
    """Extract song title from quotes."""
    if not isinstance(text, str):
        return None
    match = re.search(SONG_PATTERN, text)
    return match.group(1).strip() if match else None


def extract_artist(text):
    """Extract artist after 'by' or 'from'."""
    if not isinstance(text, str):
        return None
    m = re.search(ARTIST_PATTERN, text)
    return m.group(1).strip() if m else None


def join_text(row):
    """Combine title and body text."""
    t1 = row.get("title", "") or ""
    t2 = row.get("text", "") or ""
    return f"{t1} {t2}"


def clean_dataset(df):
    """Main cleaning pipeline."""
    df = df.copy()

    # Build combined text
    df["full_text"] = df.apply(join_text, axis=1)

    # Clean
    df["clean_text"] = df["full_text"].apply(clean_text)

    # Extract song + artist
    df["song"] = df["clean_text"].apply(extract_song)
    df["artist"] = df["clean_text"].apply(extract_artist)

    # Save cleaned version
    df.to_csv("output/cleaned_posts.csv", index=False)
    print("Saved cleaned dataset: output/cleaned_posts.csv")

    return df
