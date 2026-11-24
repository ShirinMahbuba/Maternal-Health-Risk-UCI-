# Maternal Health Risk Prediction: leakage-safe, calibrated, explainable

Classifying pregnancy risk level (low / mid / high) from six routine clinical measurements, with a focus on **how much the result can be trusted**, not only on accuracy.

**Dataset:** [UCI Maternal Health Risk](https://archive.ics.uci.edu/dataset/863/maternal+health+risk) (data collected through an IoT-based risk-monitoring system in hospitals and community clinics in rural Bangladesh, per the UCI description). Please check the UCI page for the licence and citation before publishing results.

## What this project does beyond a standard "fit a model" notebook
1. **Duplicate-leakage audit:** measures how much exact-duplicate rows inflate the score when they fall in both train and test.
2. **Honest baselines:** majority class and logistic regression next to Decision Tree, Random Forest and XGBoost.
3. **Rigorous validation:** one untouched test set, 5x5 repeated stratified CV on the training set only, model selected by CV (never by test score), bootstrap confidence interval on the final score.
4. **Clinically relevant metric:** recall of the **high-risk** class, plus macro-F1.
5. **Calibration and selective prediction:** are probabilities believable, and can the model abstain and refer uncertain cases?
6. **Explainability:** SHAP cross-checked with permutation importance.
7. **Subgroup check by maternal age** (<20, 20-34, 35+).
8. **Label-conflict check:** how many rows have identical inputs but different labels (a floor on achievable error).
9. **Transformer experiment:** FT-Transformer-style, TabTransformer-style and SAINT-style networks (small re-implementations, not the authors' code) compared with the tree models on identical CV folds.

## Results from a Colab run on the real UCI data
The notebook in this repository is saved **with its outputs** from one Colab run, so the results can be read on GitHub without running it. Figures are in `figures/`.

| Item | Result |
|---|---|
| Rows | 1014 raw, 1004 after plausibility checks, **444 unique** after removing duplicates |
| Exact duplicate rows in raw data | 562 (55.4%) |
| Random Forest macro-F1, **with** vs **without** duplicates | 0.840 vs 0.582 (test rows with an exact training copy: 81% vs 13%) |
| Label conflicts after de-duplication | 71 rows (16.0%) share identical inputs with a different label; this alone forces an error rate of at least 8.1% |
| Selected model (by CV) | XGBoost, tuned |
| Test set (n=89): accuracy / macro-F1 / high-risk recall | 0.708 / 0.622 (95% CI 0.515 to 0.728) / 0.636 |
| Baselines, test macro-F1: majority / logistic regression | 0.227 / 0.557 |
| Engineered features (MAP, pulse pressure, shock index) | no gain: macro-F1 0.630 with them vs 0.639 without |
| Sigmoid calibration | no gain: log-loss 0.725 to 0.738 |
| Top features (SHAP and permutation importance agree, Spearman 0.82) | BS, SystolicBP, BodyTemp |

**Trees vs transformer-style models** (repeated 5x5 CV on the training set, identical folds, macro-F1 mean, sd about 0.05):

| Model | Macro-F1 | High-risk recall | Mean difference vs XGBoost | Folds where it beats XGBoost |
|---|---|---|---|---|
| XGBoost | 0.630 | 0.743 | n/a | n/a |
| Decision tree | 0.616 | 0.686 | | |
| Random forest | 0.604 | 0.741 | | |
| Logistic regression | 0.550 | 0.671 | | |
| FT-Transformer-style | 0.595 | 0.785 | -0.035 | 24% |
| TabTransformer-style | 0.597 | 0.670 | -0.033 | 32% |
| SAINT-style | 0.592 | 0.762 | -0.037 | 16% |

**What this means.**
- After removing duplicates, realistic performance is modest (macro-F1 about 0.6, accuracy about 0.7), far below the 0.84 seen when duplicates are left in.
- The three transformer-style models did not beat XGBoost; each was about 0.03 to 0.04 lower in macro-F1, which is less than one standard deviation of the fold-to-fold spread. Folds overlap, so this is not a significance test. The honest reading is "no measurable benefit from transformers on this small data set", not "transformers are worse".
- The model still misses part of the high-risk group (5 of 22 high-risk test patients predicted as low risk) and does poorly on the mid-risk class.
- In the youngest group (<20) there are only 5 high-risk test cases, so that subgroup result is only indicative.

## Run in Google Colab
1. Open [colab.research.google.com](https://colab.research.google.com), then **File > Upload notebook** and choose `Maternal_Health_Risk_Prediction.ipynb`.
2. **Runtime > Run all** (about 5 to 10 minutes on Colab CPU, mostly the transformer section; faster with a GPU runtime). The data is downloaded from UCI automatically. If that fails, the notebook asks you to upload the CSV from the UCI page.
3. Figures are saved to `figures/` and offered as `figures.zip`.

## Run locally
```
pip install -r requirements.txt
jupyter notebook Maternal_Health_Risk_Prediction.ipynb
```

## Limitations 
Small single-source data set, no external validation, labels come from the data collectors rather than independent clinical outcomes, and SHAP describes the model, not causes.
