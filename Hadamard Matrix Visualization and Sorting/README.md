
# 📐 Hadamard Matrix Visualization and Sorting

## Описание

Этот проект исследует поведение матриц Адамара и предлагает метод сортировки их строк по количеству изменений знака. 
Визуализация таких отсортированных матриц позволяет наглядно наблюдать за внутренней структурой и симметрией, которая иначе была бы скрыта.

## 📁 Структура проекта

```
hadamard-visualization/
├── notebooks/
│   └── ex7_9.ipynb           # Основной исследовательский блокнот
│
├── src/
│   └── hadamard_sorter.py    # Функция сортировки матрицы Адамара
│
├── reports/
│   └── figures/
│       └── sorted_hadamard_matrix.png  # Визуализация результата
│
├── requirements.txt          # Зависимости
├── README.md                 # Описание проекта
```

## 📈 Результат

Матрица Адамара (64x64), отсортированная по числу изменений в строках, визуализирована в цветах. 
Это помогает в исследовании симметрии, плотности и особенностей чередования знаков в строках.

![Sorted Hadamard Matrix](reports/figures/sorted_hadamard_matrix.png)

## 🚀 Запуск

```bash
git clone https://github.com/your_username/hadamard-visualization.git
cd hadamard-visualization
pip install -r requirements.txt
jupyter notebook notebooks/ex7_9.ipynb
```

## 📚 Исходный код

Смотри `src/hadamard_sorter.py` для реализации сортировки.

## 🧠 Возможности развития

- Интерактивная визуализация через Plotly или Streamlit
- Применение к задачам кодирования или квантовых вычислений
- Поддержка больших размеров матриц и сохранение в CSV/PNG

## 🧾 Лицензия

MIT License
