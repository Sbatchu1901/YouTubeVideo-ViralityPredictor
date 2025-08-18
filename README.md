---
# 🎥 YouTube Video Virality Predictor

## 📌 Overview

Why do some YouTube videos gain traction quickly while others barely move?
This project explores that question by building an **end-to-end machine learning pipeline** to predict YouTube video virality using metadata, engagement metrics, and audience sentiment.

We combined **YouTube Data API** metadata with **Google Perspective API** comment toxicity scores to uncover what really drives virality.

---

## 🚀 Key Features

* **Data Collection**: Scraped 1,000+ YouTube videos (focused on "machine learning tutorial" niche) using YouTube Data API.
* **Comment Analysis**: Processed thousands of comments and measured **toxicity** using the Google Perspective API.
* **Feature Engineering**:

  * Views per day
  * Likes & comment counts
  * Sentiment & toxicity scores
  * Engagement ratios (likes/views, comments/views)
* **Modeling**: Trained a **Random Forest classifier** to classify whether a video would go viral based on a views/day threshold.
* **Explainability**: Used **SHAP values** to interpret feature importance and provide transparency into model decisions.

---

## 📊 Results

* ✅ **Accuracy**: 99.16%
* 🔥 **Top predictors**: Views/day, likes, total views, and average comment toxicity
* 🧠 **Insights**: Toxicity, surprisingly, can drive more engagement and thus influence virality

---

## 💡 Why This Matters

Most creators focus on surface-level metrics like views, but this project shows the value of combining:

* **Content performance metrics** (views, likes)
* **Audience behavior metrics** (comments, toxicity)
* **NLP analysis** of user interactions

This holistic approach helps **predict virality** and may provide actionable insights for creators and marketers.

---

## 🛠️ Tech Stack

* **Languages & Libraries**: Python, Pandas, NumPy, Scikit-learn, SHAP
* **APIs**: YouTube Data API, Google Perspective API
* **Modeling**: Random Forest Classifier, SHAP for explainability
* **Visualization**: Matplotlib, Seaborn

---

## 🔮 Future Work

* Integrating **video transcripts** for NLP-based content analysis
* Analyzing **thumbnail data** (image recognition models)
* Tracking **engagement trends over time** for time-series modeling
* Testing other algorithms like **XGBoost** or **Neural Networks** for comparison

---

## 📂 Repository Structure

```
├── data/                 # Collected YouTube metadata and comments  
├── notebooks/            # Jupyter notebooks for data exploration & modeling  
├── src/                  # Feature engineering, modeling, evaluation scripts  
├── results/              # SHAP visualizations, model results, charts  
└── README.md             # Project documentation  
```

---

## 🤝 Contributing

Got ideas to improve virality prediction? Contributions are welcome! Feel free to open an issue or submit a pull request.

---

## 🔗 Links

📄 **Project Code**: [YouTubeVideo-ViralityPredictor](https://github.com/Sbatchu1901/YouTubeVideo-ViralityPredictor)
💬 **Discussion**: [LinkedIn Post](https://www.linkedin.com/posts/srujan-kumar-batchu-17418b221_)

---


