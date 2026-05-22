# 🪐 Exoplanet Candidate Detection — Kepler Light Curve Classification

> *Can a machine learn to find planets hiding in starlight?*  
> An end-to-end ML pipeline to classify exoplanet candidates from NASA Kepler flux time series — built to handle one of the harshest class imbalances in real-world scientific data.

---

## 📌 Project Overview

Astronomers detect exoplanets by observing tiny dips in a star's brightness — a method called **transit photometry**. When a planet passes in front of its star, the light reaching us briefly drops. These dips are subtle, periodic, and buried in noise.

This project builds a complete machine learning pipeline to **automatically classify stars as exoplanet-hosting or not** using Kepler space telescope flux (brightness) time series data. The goal: save astrophysicists time by filtering out non-candidates before manual review.

---

## ⚠️ The Core Challenge — Extreme Class Imbalance

This is not a typical classification problem.

| Class | Count (Train) | Proportion |
|---|---|---|
| Non-Exoplanet (0) | ~5050 | ~98.5% |
| Exoplanet (1) | ~37 | ~1.5% |

A model that predicts "not a planet" for everything achieves **98.5% accuracy** and is completely useless. Every design decision in this pipeline was made with this imbalance in mind:

- ✅ `scale_pos_weight` in XGBoost (~134x penalty for missing a planet)
- ✅ Class weights in CNN training
- ✅ Focal loss to force the model to focus on hard, rare samples
- ✅ **PR-AUC** as the primary metric (not accuracy, not ROC-AUC)
- ✅ Threshold tuning via Precision-Recall curve rather than defaulting to 0.5
- ✅ Stratified train/val splits to preserve class ratio

Despite all of this, **overfitting remains a known limitation** of this project — with only ~37 positive examples in training, any model risks memorizing rather than generalizing. This is documented honestly in the limitations section.

---

## 🗂️ Project Structure

```
exoplanet-detection/
│
├── notebooks/
│   └── exoprocessing.ipynb        # Full pipeline notebook
│
├── raw data/
│   ├── exoTrain.csv.zip           # Kepler training set
│   └── exoTest.csv.zip            # Kepler test set
│
├── models/
│   └── Exoplanet.best.keras       # Saved best CNN model
│
└── README.md
```

---

## 🔬 Pipeline Walkthrough

### 1. Data Understanding
- **5087 training stars**, **570 test stars**
- **3197 columns**: 1 label + 3196 flux measurements over time
- Each row = one star's brightness signal over time
- No null values by design (transit data is either present or absent)

### 2. Normalization — Row-wise Z-Score
Each star is normalized **independently** using its own mean and standard deviation:

```
X_norm = (X - mean_per_star) / std_per_star
```

Why row-wise? Stars vary enormously in brightness. Global normalization would let the model cheat by learning brightness magnitude instead of **dip shape** — the actual planetary signal.

### 3. Exploratory Data Analysis
- Overlay plots of planet vs. non-planet light curves (before & after normalization)
- Average light curve per class — confirmed both classes center at 0 post-normalization
- Variance analysis to identify high-variance time steps
- Correlation heatmap of first 200 time steps

### 4. Feature Engineering (Classical ML Only)

Compressed 3196 raw flux columns → **9 statistical features**:

| Feature | Why It Matters |
|---|---|
| `mean` | Overall brightness level |
| `std` | Signal volatility |
| `min` / `max` | Extreme flux values |
| `median` | Robust center |
| `range` | max − min spread |
| `skew` | Asymmetry of the signal |
| `kurtosis` | Tail behavior / outlier presence |
| `dip_count` | Number of significant brightness dips |
| `signal_energy` | Sum of squared flux values |

> Deep learning models receive **raw normalized signals** — they extract their own features.

### 5. Models

#### XGBoost (Classical ML)
```python
XGBClassifier(
    n_estimators=500,
    max_depth=3,
    scale_pos_weight=134.62,   # Key imbalance handler
    colsample_bytree=0.7,
    subsample=0.7,
    reg_alpha=1,
    reg_lambda=1
)
```
- Threshold tuned via PR curve for best F1
- SHAP values computed for interpretability
- RandomizedSearchCV for hyperparameter tuning (20 iterations, 3-fold CV)

#### Baseline 1D CNN ⭐ Best Model
```
Input (3196 timesteps, 1 channel)
  → Conv1D(32, kernel=5, relu) → MaxPool1D(2)
  → Conv1D(64, kernel=2, relu) → MaxPool1D(2)
  → Flatten
  → Dense(128, relu) → Dropout(0.5)
  → Dense(1, sigmoid)
```
- Class weights applied during training
- Early stopping (patience=3) on val_loss
- AdamW optimizer, binary crossentropy loss

#### Tuned 1D CNN (Focal Loss)
- Added BatchNormalization after each Conv layer
- Replaced class weights with **focal loss** (`α=0.5, γ=2.0`)
- Focal loss down-weights easy negatives, forcing focus on the rare planet class

---

## 📊 Results

> Primary metric: **PR-AUC** (Precision-Recall Area Under Curve)  
> Evaluated at best F1 threshold (not default 0.5)

| Model | PR-AUC | Notes |
|---|---|---|
| XGBoost (all features) | — | Best classical ML result |
| XGBoost (selected features) | — | Slight drop vs. full features |
| **Baseline 1D CNN** | **Best** | ⭐ Final model |
| Tuned CNN (focal loss) | — | Comparable, no clear gain |

*Note: Exact metric values depend on your run. Fill in from your notebook output.*

---

## ⚙️ How to Run

### Requirements
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost tensorflow shap
```

### Steps
1. Clone the repo
2. Place `exoTrain.csv.zip` and `exoTest.csv.zip` in `raw data/`
3. Open and run `notebooks/exoprocessing.ipynb` end to end

---

## 🚧 Known Limitations

| Limitation | Details |
|---|---|
| **Severe overfitting risk** | Only ~37 positive training examples. Any model risks memorizing the training planets rather than learning generalizable transit patterns. |
| **No frequency-domain features** | FFT/periodogram features (dominant transit period, spectral energy) are absent from the classical ML pipeline — these are arguably the most informative for periodic signals. |
| **No cross-validation on CNN** | The deep learning model is validated on a single fixed split, which is fragile given the tiny positive class size. |
| **Fixed test evaluation** | Final test metrics are reported at a manually chosen threshold — small changes can shift results significantly. |

---

## 💡 Key Learnings

- **Accuracy is a lie on imbalanced data.** PR-AUC and threshold tuning matter more than any model architecture choice here.
- **Row-wise normalization is non-negotiable** for multi-source astronomical signals — global normalization destroys the signal.
- **Deep learning wins on raw time series** not because it's more powerful in general, but because it avoids the information loss from manual feature compression.
- **Focal loss is elegant but not magic.** With this little positive data, the imbalance is too extreme for any single technique to fully solve.

---

## 📚 Dataset

**NASA Kepler Exoplanet Dataset** — Labelled light curve data from the Kepler space telescope mission.  
Source: [Kaggle — Exoplanet Hunting in Deep Space](https://www.kaggle.com/datasets/keplersmachines/kepler-labelled-time-series-data)

- Label `2` = Confirmed exoplanet host star
- Label `1` = Non-exoplanet star

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7-green?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-f7931e?style=flat-square&logo=scikit-learn)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-blueviolet?style=flat-square)

---

## 👤 Author

Built as a portfolio project demonstrating end-to-end ML pipeline development on real-world scientific data with extreme class imbalance.

---

*"The universe is under no obligation to make sense to you."*  
*— Neil deGrasse Tyson*
