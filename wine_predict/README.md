# 🍷 Wine Quality Regression Project

Этот проект — попытка предсказать качество вина на основе его химических характеристик с помощью моделей машинного обучения. Задача формулируется как **регрессия**.

## 📦 Использованные этапы:

- Исследовательский анализ данных (EDA)
- Построение базовых моделей:
  - DecisionTreeRegressor
  - RandomForestRegressor
  - GradientBoostingRegressor
- Подбор гиперпараметров через GridSearchCV
- Блендинг моделей
- Стэкинг (Stacking) моделей с мета-регрессором Ridge
- Визуализация важности признаков и ошибок
- Финальный пайплайн

## 📊 Результаты

| Модель             | MAE       | MSE       | R²          |
| ------------------ | --------- | --------- | ----------- |
| RandomForest       | 0.428     | 0.353     | 0.522       |
| GradientBoosting   | 0.408     | 0.361     | 0.510       |
| XGBoost            | 0.44    | 0.37   | 0.49      |
| **Blending (Avg)** | **0.418** | **0.349** | **0.527** ✅ |

> Лучшая модель: стекинг из RandomForest, GradientBoosting, XGBoost  
> Мета-модель: Ridge  
> **R² ≈ 0.53**

## 📈 Используемые библиотеки

- pandas, numpy, matplotlib, seaborn
- scikit-learn
- xgboost

## 📚 Источник данных

[UCI ML Repository – Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)

## 📝 License

Проект открыт под лицензией MIT.

