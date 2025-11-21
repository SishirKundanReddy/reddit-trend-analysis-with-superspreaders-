

# 🎵 Reddit Music Trend Analysis

### *A Social Computing + Machine Learning Project*

This project analyzes **how music trends emerge on Reddit**.
It collects data from major music subreddits, cleans and processes the text, performs sentiment analysis on comments, detects “superspreaders” (influential users), and trains machine learning models to predict whether a post is **trending**, **emerging**, or **low-engagement**.

It is designed as a complete **end-to-end research pipeline** suitable for academic use (B.Tech projects), social computing studies, or trend-prediction experiments.

---

## 🚀 Features

### **1. Reddit Data Collection**

Fetches posts from multiple music-related subreddits using Reddit’s public JSON API:

* `r/Music`
* `r/PopHeads`
* `r/HipHopHeads`
* `r/indieheads`

Extracts for each post:

* Title
* Score (upvotes)
* Number of comments
* Post timestamp
* URL
* Raw + cleaned text
* Automatically detected **song** and **artist** names

---

### **2. Cleaning + Trend Analysis**

Cleans text (removing noise, lowercasing, etc.) and computes:

* Trend keywords (e.g., “new”, “leaked”, “breaking”, “out now”)
* Engagement score
* Engagement label:

  * **TRENDING**
  * **EMERGING**
  * **STABLE**
  * **LOW**

---

### **3. Sentiment Analysis**

Scrapes comments from each post and performs **VADER sentiment analysis**:

* Positive / Negative / Neutral scores
* Compound sentiment score
* Final sentiment label

This helps measure the *reaction* to songs, artists, news, and releases.

---

### **4. Machine Learning Model**

Trains ML models to predict a post’s trend status:

* **Random Forest** (primary model)
* **Logistic Regression**

Outputs:

* Accuracy score
* Precision, recall, F1-score
* Saved model: `output/model.pkl`

---

### **5. Superspreader Detection (Network Analysis)**

Builds a **user interaction graph** using:

* Post author ↔ commenter relationships
* Co-commenter relationships

Computes:

* **Degree centrality**
* **Betweenness centrality**
* **PageRank**

Identifies the most influential users (“superspreaders”) who amplify music discussions.
Outputs:

* `output/superspreaders.csv`
* `output/charts/superspreaders_graph.png`

---

### **6. Visualizations**

Automatically generates a set of insightful charts:

* Sentiment distribution
* Most mentioned artists / songs
* Subreddit score comparisons
* Engagement category visualization
* Trend keyword frequency
* Score vs. comment scatterplot
* Feature correlation heatmap
* Superspreader network graph

All charts are saved in `output/charts/`.

---

## 🧠 Project Structure

```
reddit_music_trends/
│
├── main.py
├── modules/
│   ├── reddit_fetch.py
│   ├── data_cleaning.py
│   ├── trend_analysis.py
│   ├── sentiment_analysis.py
│   ├── ml_model.py
│   ├── visualization.py
│   └── superspreaders.py
│
├── output/
│   ├── raw_posts.csv
│   ├── cleaned_posts.csv
│   ├── trend_dataset.csv
│   ├── reddit_comment_sentiment.csv
│   ├── superspreaders.csv
│   ├── model.pkl
│   └── charts/
│
└── README.md
```

---

## ▶️ How to Run

### **1. Install dependencies**

```bash
pip install -r requirements.txt
```

### **2. Run the full pipeline**

```bash
python main.py
```

All data, sentiment, superspreader results, ML models, and visualizations will appear in the `output/` folder.

---

## 📌 Academic Relevance

This project is suitable for:

* B.Tech / B.S. Social Computing Coursework
* Machine Learning & NLP Projects
* Trend Prediction Research
* Network Analysis / Centrality Projects
* Final-year project portfolios
* GitHub showcase projects

---

## 👨‍💻 Author

**P. Sishir Kundan Reddy**
B.Tech, 3rd Year

* **LinkedIn:** [https://www.linkedin.com/in/sishirkundan-reddy-3320b22b7](https://www.linkedin.com/in/sishirkundan-reddy-3320b22b7)
* **GitHub:** [https://github.com/SishirKundanReddy](https://github.com/SishirKundanReddy)

---

