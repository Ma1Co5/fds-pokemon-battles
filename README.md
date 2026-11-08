# FDS: Pokémon Battles Prediction 2026

Code for our three final submissions to the Kaggle challenge
*FDS: Pokémon Battles Prediction 2026* (Fundamentals of Data Science, Sapienza University of Rome).

| Script | Submission |
|---|---|
| `src/model_final.py` | 1 – final model: hidden Player 2 knock-outs (Bayes), Player 2 team reconstruction, out-of-fold encoders, soft-voting ensemble of logistic regression and two `HistGradientBoostingClassifier` |
| `src/model_advanced_features.py` | 2 – base, extended and advanced features; logistic regression or Random Forest, chosen by 5-fold CV |
| `src/model_base_features.py` | 3 – base and extended features; logistic regression or Random Forest, chosen by 5-fold CV |

## How to run

The scripts look for the competition data (`train.jsonl`, `test.jsonl`, `type_chart_gen2.csv`,
`sample_submission.csv`) under `/kaggle/input`, or in the current folder when run elsewhere.

```bash
python src/model_final.py submission_1_final.csv
python src/model_advanced_features.py submission_2_advanced_features.csv
python src/model_base_features.py submission_3_base_features.csv
```

Only the libraries of the standard Kaggle Python image are needed (numpy, pandas, scipy, scikit-learn).
All random seeds are fixed, so each script reproduces its submission.
The Kaggle notebook of the team clones this repository and runs the three scripts.
