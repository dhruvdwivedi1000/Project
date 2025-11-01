# Titanic Machine Learning Pipeline

This repository contains a complete end-to-end pipeline for solving the Titanic survival prediction challenge.

## Contents
- **Titanic.ipynb** → Jupyter Notebook implementing the pipeline step by step.
- **README.md** → Documentation and instructions.

## Pipeline Overview
1. **Problem framing:** Binary classification (Survived 0/1).
2. **Data loading & exploration:** Missingness, target balance, basic stats.
3. **Feature engineering:** Title extraction, FamilySize, IsAlone, TicketGroupSize, FarePerPerson, Deck, missingness indicators.
4. **Preprocessing:** Imputation, scaling, one-hot encoding using `ColumnTransformer`.
5. **Modeling:** Logistic Regression, Random Forest, Gradient Boosting, LightGBM.
6. **Cross-validation:** Stratified K-Fold, accuracy + ROC AUC metrics.
7. **Evaluation:** Confusion matrix, precision/recall/F1, feature importances, SHAP explanations.
8. **Submission:** Generates `submission.csv` for Kaggle.

## Requirements
Install dependencies with:
```bash
pip install -r requirements.txt
```
Main packages used:
- numpy, pandas, matplotlib, seaborn
- scikit-learn
- lightgbm (optional, for boosted trees)
- shap (optional, for explainability)

## Usage
1. Open `Titanic.ipynb` in Jupyter or VS Code.
2. Run cells in order.
3. The notebook will save a Kaggle-compatible `submission.csv` file with predictions.

## Next Steps
- Try CatBoost or XGBoost as alternative models.
- Use SHAP for model interpretability.
- Deploy the model as a small web app with Streamlit or Flask.

---

**Author:** Auto-generated guide with enhancements for reproducibility and clarity.
