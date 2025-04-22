# 🧬 Breast Cancer Classification & ROC AUC Visualization

This project is a machine learning exploration using breast cancer data to train and compare multiple classification models.  
It also includes an animated visualization of how ROC AUC evolves during training of an XGBoost model.

---

## 📊 Project Overview

- **Dataset**: Breast cancer dataset (4024 samples)
- **Goal**: Predict patient status (Alive / Dead)
- **Features**: Clinical and biological data (tumor size, hormone status, stage, etc.)
- **Challenge**: Class imbalance (handled via `RandomOverSampler`)

---

## ✅ What’s Included

- Preprocessing with label encoding and correlation-based feature selection
- Cross-validation on multiple models:
  - Logistic Regression
  - Random Forest
  - SVM
  - Gradient Boosting
  - XGBoost
- ROC AUC metric tracking and comparison
- Interactive animated visualization of ROC AUC during XGBoost training
- Export to `.html` and `.gif` format

---


## 📦 Dependencies



Main libraries:
- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `plotly`
- `imbalanced-learn`
- `kaleido` *(for exporting Plotly figures to PNG)*
- `imageio` *(for gif creation)*

---

## 🚀 Try It Yourself

You can run this notebook in:
- Jupyter Notebook or Jupyter Lab
- Google Colab *(upload the notebook and dataset)*

---

## 📚 Future Improvements

- Add train/test split and evaluate on hold-out set
- Include more models (e.g., LightGBM, CatBoost)
- Use Optuna or GridSearchCV for hyperparameter tuning

---

## 👨‍💻 Author

Made with ❤️ by [ww_pierre]
