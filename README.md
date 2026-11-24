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

All numbers are printed by the notebook; none are hard-coded in this README. Run it to get the results.

## Run in Google Colab
1. Open [colab.research.google.com](https://colab.research.google.com), then **File > Upload notebook** and choose `Maternal_Health_Risk_Prediction.ipynb`.
2. **Runtime > Run all** (about 1 to 2 minutes). The data is downloaded from UCI automatically. If that fails, the notebook asks you to upload the CSV from the UCI page.
3. Figures are saved to `figures/` and offered as `figures.zip`.

## Run locally
```
pip install -r requirements.txt
jupyter notebook Maternal_Health_Risk_Prediction.ipynb
```

## Limitations (state these in any write-up)
Small single-source data set, no external validation, labels come from the data collectors rather than independent clinical outcomes, and SHAP describes the model, not causes.
