# 🌌 Exoplanet Prediction Model

This project builds a **machine learning pipeline** to predict whether a Kepler Object of Interest (KOI) is a **Confirmed Exoplanet** or a **False Positive** using NASA Kepler mission data.  

The solution is end-to-end: it covers data preprocessing, model training, saving, prediction on new data, and retraining for updated datasets.

---

## 📖 Table of Contents
1. [Introduction](#introduction)  
2. [Dataset](#dataset)  
3. [Preprocessing](#preprocessing)  
4. [Model Development](#model-development)  
5. [Training and Evaluation](#training-and-evaluation)  
6. [Pipeline and Model Saving](#pipeline-and-model-saving)  
7. [Prediction on New Data](#prediction-on-new-data)  
8. [Retraining](#retraining)  
9. [Results and Outputs](#results-and-outputs)  
10. [Future Work](#future-work)  
11. [Acknowledgements](#acknowledgements)  
12. [License](#license)  

---

## 🛰 Introduction

The Kepler Space Telescope identified thousands of potential exoplanets. However, not all detections are true planets — some are **false positives** caused by binary stars, noise, or instrumentation artifacts.  

The goal of this project is to automate the classification process using **machine learning**, achieving high accuracy while being scalable and reusable.  

---

## 📂 Dataset

- **Source**: [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)  
- **Format**: Tabular CSV file containing ~49 columns per KOI.  
- **Target Variable**: `label`  
  - `1` → Confirmed Exoplanet  
  - `0` → False Positive  

### Example Raw Data (before preprocessing):

| kepid    | kepoi_name | kepler_name | koi_disposition | koi_fpflag_nt | koi_fpflag_ss | koi_period | koi_prad | koi_teq | koi_steff | koi_slogg | koi_srad | koi_kepmag | ... |
|----------|------------|--------------|-----------------|---------------|---------------|------------|----------|---------|-----------|-----------|----------|------------|-----|
| 10797460 | K00752.01  | Kepler-227 b | CONFIRMED       | 0             | 0             | 15.3       | 2.3      | 821     | 5578      | 4.46      | 0.92     | 15.3       | ... |

---

## ⚙️ Preprocessing

Steps taken before training:

1. **Feature Selection** → Out of 49 features, only 16 were relevant for the model:

2. **Scaling** → Applied `StandardScaler` to normalize numerical features.  

3. **Automation** → Built into a `ColumnTransformer` so the pipeline automatically drops unused columns and scales selected ones.

---

## 🤖 Model Development

Several models were tested:

- **Logistic Regression** → Baseline, fast but limited.  
- **Random Forest Classifier** 🌲 → Best performance (~99.4% accuracy).  
- **XGBoost / LightGBM** → Future candidates for improvement.  

Final choice: **Random Forest with class balancing**.  

---

## 🏋️ Training and Evaluation

Split: **80% Training / 20% Testing**  

### Logistic Regression Results:
- Accuracy: 98.4%  
- Precision: 97.1%  
- Recall: 98.5%  
- F1: 97.8%  

### Random Forest Results:
- Accuracy: **99.4%**  
- Precision: 99.6%  
- Recall: 98.9%  
- F1: 99.2%  

**Confusion Matrix (Random Forest):**

✅ Random Forest was chosen as the final model.

---

## 💾 Pipeline and Model Saving

To ensure reproducibility and easy deployment:

- Used `Pipeline` to combine preprocessing + model:
  ```python
  pipeline = Pipeline([
      ("preprocessor", preprocessor),
      ("rf", RandomForestClassifier(...))
  ])
```python
import joblib, pandas as pd

pipeline = joblib.load("rf_pipeline.pkl")
df = pd.read_csv("new_input.csv")

predictions = pipeline.predict(df)
## 🔮 Prediction on New Data

To test on unseen KOI data:

```python
import joblib, pandas as pd

pipeline = joblib.load("rf_pipeline.pkl")
df = pd.read_csv("new_input.csv")

predictions = pipeline.predict(df)
df["prediction"] = ["CONFIRMED" if p == 1 else "FALSE POSITIVE" for p in predictions]
print(df[["kepid","kepoi_name","kepler_name","prediction"]].head())
```



