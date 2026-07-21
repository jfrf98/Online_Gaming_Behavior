# Online Gaming Behavior

Multiclass machine learning project designed to predict player engagement levels from demographic and in-game behavioral data. The analysis explores how play frequency, session duration, progression, achievements, purchases, and player characteristics relate to **Low**, **Medium**, and **High** engagement.

## Business context

Understanding player engagement can help gaming teams identify behavioral patterns associated with highly engaged users and support decisions related to:

- player retention and re-engagement strategies;
- game design and difficulty balancing;
- personalized experiences and player segmentation;
- prioritization of product and marketing initiatives.

The model is intended as an analytical proof of concept. It predicts an engagement category; it does not establish that a particular behavior causes engagement.

## Project objective

Build and compare multiclass classification models capable of predicting `engagement_level` while answering three practical questions:

1. Can player engagement be predicted more effectively than a majority-class baseline?
2. Which model offers the strongest balance of discrimination, classification performance, and training time?
3. How could the resulting predictions support product and retention decisions?

## Dataset

The dataset contains **40,034 player records** and **13 original variables**.

| Variable group | Examples |
|---|---|
| Demographics | `age`, `gender`, `location` |
| Game context | `game_genre`, `game_difficulty` |
| Activity | `play_time_hours`, `sessions_per_week`, `avg_session_duration_minutes` |
| Progression | `player_level`, `achievements_unlocked` |
| Monetization | `in_game_purchases` |
| Target | `engagement_level` (`Low`, `Medium`, `High`) |

The target is moderately imbalanced: approximately **48.4% Medium**, **25.9% High**, and **25.7% Low** in the training sample.

## Methodology

### 1. Data exploration and preparation

- Inspected data types, descriptive statistics, and feature distributions.
- Standardized column names using `snake_case`.
- Reviewed numerical correlations and categorical distributions.
- Split the data into 80% training and 20% testing sets.
- Standardized numerical variables using `StandardScaler`.
- Encoded categorical variables using `OneHotEncoder` with unknown-category handling.

### 2. Feature engineering

Five behavioral features were created:

- `weekly_playtime`
- `playtime_per_level`
- `achievements_per_level`
- `sessions_per_hour`
- `achievement_progress`

These features combine activity and progression signals to represent player behavior more directly than the raw variables alone.

### 3. Model comparison

The following classifiers were evaluated:

- Majority-class `DummyClassifier`
- Logistic Regression
- Decision Tree
- Random Forest
- CatBoost
- XGBoost

Performance was assessed with accuracy, weighted precision, weighted recall, weighted F1-score, multiclass ROC-AUC (one-vs-rest), confusion matrices, and training time.

## Results

### Best model by ROC-AUC

**XGBoost** with `max_depth=5`, `learning_rate=0.05`, and `n_estimators=300` achieved the highest test ROC-AUC.

| Model | Accuracy | Weighted precision | Weighted recall | Weighted F1 | ROC-AUC OvR | Training time* |
|---|---:|---:|---:|---:|---:|---:|
| Dummy baseline | 0.4845 | — | 0.4845 | 0.3162 | — | — |
| Logistic Regression (`C=5`) | 0.8303 | 0.8429 | 0.8303 | 0.8314 | 0.9327 | 0.19 s |
| Decision Tree (`depth=7`) | 0.9156 | 0.9155 | 0.9156 | 0.9155 | 0.9390 | 0.43 s |
| Random Forest (`n=300`, unrestricted depth) | 0.9208 | 0.9208 | 0.9208 | 0.9206 | 0.9434 | 13.38 s |
| CatBoost (`depth=6`, `lr=0.05`) | 0.9232 | 0.9232 | 0.9232 | 0.9230 | 0.9441 | 2.05 s |
| **XGBoost (`depth=5`, `lr=0.05`)** | **0.9232** | **0.9231** | **0.9232** | **0.9230** | **0.9449** | **1.48 s** |

\*Training times were measured in the local notebook environment and may vary by hardware and software version.

The selected XGBoost configuration improved accuracy by approximately **43.9 percentage points** and weighted F1 by approximately **60.7 percentage points** over the majority-class baseline.

CatBoost produced the highest observed accuracy in the experiment (**0.9238** with `depth=8`, `learning_rate=0.05`), but XGBoost achieved the strongest ROC-AUC and trained faster. This illustrates why model selection should consider several metrics rather than accuracy alone.

## Business interpretation

The results suggest that player behavior contains enough signal to distinguish engagement levels with strong test performance. In practice, model scores could be used to:

- identify players at risk of low engagement for targeted re-engagement;
- segment users for personalized content or difficulty recommendations;
- prioritize experiments around session frequency, duration, and progression;
- monitor changes in the predicted engagement mix after product updates.

Before operational use, the model should be validated on out-of-time or live production data and evaluated against the cost of each type of classification error.

## Technologies

- Python
- pandas and NumPy
- Matplotlib and Seaborn
- scikit-learn
- XGBoost
- CatBoost
- SciPy
- Jupyter Notebook

## How to run the project

1. Clone the repository and create a virtual environment.

   ```bash
   python -m venv .venv
   ```

2. Activate the environment.

   ```bash
   # Windows
   .venv\Scripts\activate

   # macOS / Linux
   source .venv/bin/activate
   ```

3. Install the dependencies.

   ```bash
   pip install pandas numpy matplotlib seaborn scipy scikit-learn xgboost catboost jupyter
   ```

4. Place `online_gaming_behavior_insights.csv` in a local `data/` directory and use a relative path in the notebook:

   ```python
   df = pd.read_csv("data/online_gaming_behavior_insights.csv")
   ```

5. Start Jupyter and run the notebook from top to bottom.

   ```bash
   jupyter notebook
   ```

## Limitations and next steps

- Remove `player_id` from the feature matrix because it is an identifier and should not contribute to prediction.
- Use a stratified train/test split and cross-validation for more reliable model comparison.
- Move preprocessing and modeling into a single scikit-learn pipeline to reduce leakage and deployment risk.
- Reserve the test set for final evaluation instead of using it repeatedly during hyperparameter selection.
- Correct the feature-name mapping used for model-importance analysis after one-hot encoding.
- Add per-class precision, recall, and F1 to verify performance beyond weighted averages.
- Calibrate predicted probabilities and define business-specific decision thresholds.
- Validate the model on new or time-separated player data before drawing operational conclusions.

## Key takeaway

Tree-based ensemble models substantially outperformed the baseline and Logistic Regression. XGBoost provided the best ROC-AUC with a strong balance of predictive performance and training speed, making it the leading candidate for further validation and deployment-oriented development.

---

Developed as a portfolio project focused on exploratory data analysis, feature engineering, multiclass classification, and business-oriented model evaluation.
