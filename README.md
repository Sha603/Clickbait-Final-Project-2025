# Clickbait Detection Using Advanced Text Vectorization Techniques

## Overview
This project focuses on **detecting clickbait headlines** using smart machine learning models and **advanced text vectorization techniques**. Clickbait headlines are tricky titles designed to grab attention but often mislead readers. Here, we **analyze different ways** to transform text into numbers (TF-IDF, Word2Vec, Doc2Vec) and use machine learning models to **identify clickbait** automatically.

## Key Highlights
- **Text Vectorization**: We tested three methods:
  - **TF-IDF**: Counts how important each word is.
  - **Word2Vec**: Understands the meaning and context of words.
  - **Doc2Vec**: Represents entire headlines as a single vector.
- **Machine Learning Models**:
  - Support Vector Machine (SVM)
  - Logistic Regression (LR)
  - Random Forest (RF)
  - XGBoost
  - Gradient Boosting (GB)
- **Winner**:
  - **Word2Vec + XGBoost** achieved the best results!
  - Reached **95%** in Accuracy, Precision, Recall, and F1-Score.

## Project Objective
- Detect misleading clickbait headlines.
- Compare different vectorization techniques.
- Find the best ML model to automate clickbait detection.

## Dataset
- **Source**: Kaggle Clickbait Dataset.
- **Size**: 32,000 headlines (50% clickbait, 50% non-clickbait).
- **Sources**:
  - Clickbait examples from BuzzFeed, Upworthy, etc.
  - Non-clickbait from New York Times, WikiNews, etc.

## Approach
### 1. Data Preprocessing
- Remove noise (punctuation, URLs).
- Apply **stemming** and **lemmatization**.
- Remove common stopwords like “the”, “is”, “and”.

### 2. Vectorization Techniques
- **TF-IDF**: Highlights important words.
- **Word2Vec**: Captures the meaning between words.
- **Doc2Vec**: Embeds full sentences.

### 3. Machine Learning Models
- Train SVM, LR, RF, XGBoost, and GB on vectorized data.
- Optimize hyperparameters with **GridSearchCV**.

### 4. Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score

## Results Summary
| Vectorizer | Best Model | Accuracy | F1-Score |
|------------|------------|----------|----------|
| TF-IDF     | SVM, LR    | 94%      | 94%      |
| Word2Vec   | XGBoost    | 95%      | 95%      |
| Doc2Vec    | SVM        | 91%      | 91%      |

- **Word2Vec** consistently captured clickbait nuances better.
- **Doc2Vec** performed well but slightly weaker.
- **TF-IDF** performed strongly with traditional models.

## Tools & Technologies Used
- Python
- NLTK (for preprocessing)
- Scikit-learn (for modeling)
- Gensim (for Word2Vec and Doc2Vec)
- XGBoost library
- Matplotlib, Seaborn (for visualization)

## How to Run the Project
1. Install necessary libraries:
```bash
pip install numpy pandas scikit-learn gensim xgboost matplotlib seaborn nltk
```
2. Preprocess the data:
```bash
python preprocess_data.py
```
3. Train the models:
```bash
python train_models.py
```
4. Evaluate results:
```bash
python evaluate_models.py
```

## Challenges Faced
- **Data Imbalance**: Luckily, the dataset was balanced (50/50 split).
- **Semantic Complexity**: Clickbait words often use emotional triggers that simple models miss.
- **Hyperparameter Tuning**: Required extensive optimization for best results.

## Future Work
- Use **transformer models** like BERT for even better accuracy.
- **Multilingual Clickbait Detection** (currently English-only).
- Develop a **real-time detection system** (like a browser plugin).
- Create hybrid vectorization (combining Word2Vec + TF-IDF).

## Conclusion
✅ Word2Vec + XGBoost is the best combination for detecting clickbait!  
✅ The project successfully builds a solid system that can assist news websites, social media platforms, and even readers in avoiding misleading headlines.


