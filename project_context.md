# 🎮 Game Popularity Prediction (Model Building Phase)

### Authors

* **Shane Calder (Undergraduate)**
* **Duy Nguyen (Undergraduate)**

---

## 📘 Overview

This project predicts the **popularity of a video game** using a combination of **metadata** and **text descriptions** from the Steam Games Dataset.

Popularity is measured by **Estimated Owners**, which serves as our **target variable**.

This phase focuses on **machine learning model training**, **feature selection**, and **model saving**, building on the preprocessing work completed earlier.

---

## 📊 Dataset

* **Source:** [Steam Games Dataset – Hugging Face](https://huggingface.co/datasets/FronkonGames/steam-games-dataset)
* **Datapoints:** 83,560
* **Features:** 39
* **Target Variable:** Estimated Owners

### Key Attributes

* Price
* Categories / Genres
* DLC Count
* Expected Playtime
* Review Counts & Recommendations
* Player Activity Metrics
* “About the Game” text descriptions

### Preprocessing Summary

The dataset was fully preprocessed in the previous phase:

* Cleaned and normalized numeric data
* Encoded categorical variables
* Scaled continuous features
* **Applied NLP preprocessing using SentenceTransformer** to embed “About the Game” text

The resulting dataset combines structured metadata and NLP embeddings into a **hybrid feature set** for model training.

---

## ⚙️ Model Building Tasks

### 1. Model Development

* Use the preprocessed dataset (metadata + NLP embeddings).
* Build at least one regression model (e.g., Linear Regression, Random Forest, XGBoost) to predict *Estimated Owners*.

### 2. Feature Selection

* Identify and select the most relevant predictors.
* Build two versions of the model:

  * **Model 1:** Full feature set (all metadata + NLP features)
  * **Model 2:** Reduced feature set (selected subset)

### 3. Model Training & Saving

* Split data into training and testing sets.
* Train both models.
* Save trained models as `.sav` files:

```python
import pickle
filename = 'finalized_model_M1.sav'
pickle.dump(model, open(filename, 'wb'))

# Load and test
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)
print(result)
```

### 4. Deliverables

* Jupyter Notebook showing:

  * Feature selection steps
  * Model training for full & reduced features
  * Model saving/loading process
* `.sav` model files for each trained model

---

## 🧩 Project Status

* ✅ **Preprocessing complete** (including NLP embeddings)
* ✅ **SentenceTransformer integrated** for text representation
* 🚧 **Current phase:** Train and save two regression models
* ⏳ **Next phase:** Expand to 4+ ML models and include performance metrics

---

## 🧠 Technical Details

* **Task Type:** Regression (predicting Estimated Owners)
* **Feature Composition:** Hybrid (structured + NLP embeddings)
* **Libraries:** `pandas`, `scikit-learn`, `numpy`, `pickle`, `matplotlib`, `seaborn`, `sentence-transformers`
* **Outputs:** `.sav` model files + training notebook
