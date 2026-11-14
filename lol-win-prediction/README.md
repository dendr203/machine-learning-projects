# League of Legends Win Prediction at 15 Minutes

Binary classification of ranked League of Legends matches: given team statistics at minute 15, predict whether the blue side wins. The notebook covers exploratory analysis, data-quality fixes, preprocessing, and a tuned comparison of four models (Decision Tree, Random Forest, Logistic Regression, XGBoost). The best model reaches **75.2% test accuracy and 0.834 ROC AUC**.

## Problem

By minute 15 a League of Legends match has built up a measurable state: gold, experience, kills, objectives and vision. How much of the final result is already decided at that point?

- **Live win probability.** A calibrated model of this kind is the core of in-game win-probability overlays for broadcasts and post-game analytics.
- **Game balance and design insight.** Measuring how strongly early leads turn into wins shows how "snowbally" the game is, and how much the early game decides the outcome compared with mid and late game.
- **Player feedback.** Seeing which early-game stats separate wins from losses helps players prioritize.

## Dataset

- **Source:** [League of Legends SoloQ matches at 15 minutes 2024](https://www.kaggle.com/datasets/karlorusovan/league-of-legends-soloq-matches-at-10-minutes-2024/data) (Kaggle, user `karlorusovan`)
- **Scope:** 2024 ranked solo-queue matches (mostly EUW and EUNE servers), Emerald to Diamond
- **Size:** 24,225 matches; 23,685 after cleaning
- **Features (27):** for each team: wards placed, control wards placed, kills, dragons, Rift Heralds, towers, inhibitors, turret plates, lane minions, jungle minions, total gold, total XP, damage to champions. Plus `blueTeamFirstBlood`.
- **Target:** `blueWin` (1 = blue side wins). The classes are balanced, with the blue side winning about 49.5% of matches.

## Approach

1. **EDA:** feature distributions, class balance and a full correlation matrix.
2. **Data cleaning:**
   - The raw turret-plate columns were swapped between teams: `blueTeamTurretPlatesDestroyed` actually counted plates the blue team *lost*. The columns were relabeled so each one counts plates the team destroyed.
   - Rows with impossible values were removed: more than 15 plates destroyed by a team (3 outer towers × 5 plates), or more than 2 Herald kills.
   - Dropped `matchID` and a trailing empty column.
3. **Split:** stratified 80/20 train/test split (18,948 / 4,737 matches), `random_state=42`.
4. **Scaling:** a `ColumnTransformer` with `StandardScaler` for roughly Gaussian features (gold, XP, kills, minions, damage, plates) and `RobustScaler` for skewed, outlier-heavy features (wards, towers, inhibitors). It is fitted on the training set only.
5. **Models:** Decision Tree, Random Forest, Logistic Regression and XGBoost, each trained first with default settings and then tuned.
6. **Tuning:** `RandomizedSearchCV` (30 iterations, 5-fold CV, accuracy scoring) over each model's main hyperparameters.
7. **Evaluation:** accuracy, per-class precision/recall/F1, confusion matrix, ROC curve and AUC on the held-out test set.

## Results

Held-out test set (4,737 matches). CV accuracy is the mean 5-fold score on the training set.

| Model               | Baseline accuracy | Baseline AUC | Tuned accuracy | Tuned AUC | Tuned CV accuracy |
|---------------------|------------------:|-------------:|---------------:|----------:|------------------:|
| Decision Tree       | 0.669             | 0.669        | 0.726          | 0.801     | 0.740             |
| Random Forest       | 0.741             | 0.825        | 0.747          | 0.830     | 0.759             |
| **Logistic Regression** | **0.752**     | **0.834**    | 0.752          | 0.833     | 0.760             |
| XGBoost             | 0.736             | 0.812        | 0.747          | 0.831     | **0.761**         |

Best tuned hyperparameters:
- **Logistic Regression:** `C=0.01`, `penalty='l2'`, `solver='saga'`, `max_iter=1000`
- **XGBoost:** `n_estimators=100`, `max_depth=3`, `learning_rate=0.1`, `subsample=1.0`, `colsample_bytree=0.8`, `reg_lambda=1`, `reg_alpha=0.1`, `gamma=0`
- **Random Forest:** `n_estimators=200`, `max_depth=10`, `max_features='sqrt'`, `min_samples_leaf=4`, `criterion='entropy'`, `bootstrap=False`
- **Decision Tree:** `max_depth=5`, `min_samples_split=20`, `min_samples_leaf=4`, `criterion='gini'`

## Key Insights

- **Simple beats complex here.** Untuned logistic regression beats tuned Random Forest and XGBoost on the test set (0.752 vs. 0.747 accuracy, 0.834 vs. 0.830/0.831 AUC). Tuned XGBoost has a marginally higher CV accuracy (0.761 vs. 0.760), so the top three models are practically tied. The strong showing of a linear model suggests that the 15-minute signal is largely additive: each extra unit of gold, XP or objective control shifts the odds roughly independently of the others.
- **A ~75% ceiling.** Every tuned model except the decision tree levels off at about 75% accuracy (AUC 0.83). The state at minute 15 is strongly predictive, but about a quarter of games are still decided later, through teamfights, scaling compositions and late objectives.
- **Gold, XP, damage and kills move together.** The correlation matrix shows that a team ahead in resources deals more damage and gets more kills, which reinforces the lead.
- **Turret plates are a key early gold source.** Team gold correlates with towers and plates destroyed. Plates grant bonus gold until they fall off at minute 14.
- **Dragons are zero-sum.** Blue and red dragon counts are strongly negatively correlated, because every dragon one team takes is denied to the other.
- **Single trees overfit.** An unconstrained decision tree reaches only 0.669 accuracy. Limiting its depth to 5 raises AUC from 0.669 to 0.801.

## How to Run

```bash
git clone https://github.com/dendr203/machine-learning-projects.git
cd machine-learning-projects/lol-win-prediction
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook lol_win_prediction.ipynb
```

The project lives in the `lol-win-prediction/` subfolder of the repository. Run all cells from top to bottom. The notebook reads the data from `data/match_data_v5.csv`, relative to that folder.

The results above were produced with pandas 3.x, scikit-learn 1.9.1 and xgboost 3.4.1. Other versions, XGBoost in particular, may give slightly different numbers.

## File Structure

```
lol-win-prediction/
├── data/
│   └── match_data_v5.csv        # Raw match data (Kaggle)
├── lol_win_prediction.ipynb     # EDA, cleaning, modeling and evaluation
├── requirements.txt
└── README.md
```

## Possible Next Steps

- Feature importance analysis (logistic regression coefficients, SHAP for XGBoost) to rank which early objectives matter most
- Team-difference features (e.g. gold difference, XP difference) instead of raw per-team totals
- Probability calibration and log-loss evaluation for a live win-probability use case
- Snapshots at several timestamps (10, 15, 20 minutes) to track how predictability grows over a game
