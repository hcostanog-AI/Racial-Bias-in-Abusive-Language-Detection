# ⚖️ Racial Bias in Abusive Language Detection
*Replication of Davidson et al. (2017) using Logistic Regression & TF-IDF*

---

## 📖 Overview
This project replicates part of the study by **Davidson, Bhattacharya & Weber (2017)**, which demonstrated racial bias in abusive language detection datasets.  
The original research showed that tweets written in **African-American English (AAE)** were disproportionately classified as abusive compared to **Standard American English (SAE)**.  

By re-implementing their method, this project highlights the risks of bias in AI systems for content moderation.

---

## 🛠️ Methodology
- **Dataset:** Davidson et al. (2017), ~24k tweets labeled as *Hate*, *Offensive*, or *Neither*.  
- **Preprocessing:**
  - URL & mention replacement with placeholders.
  - Tokenization (`TweetTokenizer`) & stemming (`SnowballStemmer`).
  - TF-IDF feature extraction with unigrams, bigrams, trigrams (10,000 features).  
- **Model:** Logistic Regression with stratified 5-fold cross-validation.  
- **Hyperparameter Tuning:** Grid search over regularization strength.  

---

## 📊 Results
| Class      | F1 (Original Study) | F1 (Replication) |
|------------|---------------------|------------------|
| Hate       | 0.40                | 0.43             |
| Offensive  | 0.92                | 0.92             |
| Neither    | 0.87                | 0.81             |

**Key Insight:** Both the original and replicated results show that the classifier struggles with *Hate Speech* detection while performing well on more frequent classes (*Offensive*, *Neither*).  
This imbalance underscores how dataset bias and class imbalance can distort model outcomes — with real risks of **discriminatory enforcement**.

---

## 🔍 Critical Reflection
- Logistic Regression was chosen for **interpretability**, isolating dataset bias rather than model complexity.  
- Bias likely stems from:
  - Class imbalance (few *Hate* examples).  
  - Misannotation of AAE tweets due to annotator bias.  
  - Reliance on surface-level n-grams rather than semantic meaning.  
- Fairness metrics beyond F1 (e.g., equal opportunity difference) could enhance the diagnostic power.  

---

## 🧰 Tech Stack
- Python  
- pandas, numpy  
- scikit-learn  
- nltk  

---

## 📚 Reference
Davidson, T., Bhattacharya, D., & Weber, I. (2017). *Racial bias in hate speech and abusive language detection datasets.* [arXiv:1905.12516](https://arxiv.org/abs/1905.12516)

---
