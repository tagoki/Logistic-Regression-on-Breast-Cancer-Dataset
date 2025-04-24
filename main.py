import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

# Загружаем данные
x, y = load_breast_cancer(return_X_y=True)

# Делим данные на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=43)

# Масштабируем признаки
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Обучаем модель
model = LogisticRegression(max_iter=10000)
model.fit(X_train_scaled, y_train)

# Предсказания
predict_model_x_train = model.predict(X_train_scaled)
predict_model_x_test = model.predict(X_test_scaled)

# Точность
print("Train Accuracy:", accuracy_score(y_train, predict_model_x_train))
print("Test Accuracy:", accuracy_score(y_test, predict_model_x_test))

# 🔁 Кросс-валидация на всех данных (x и y)
# Предварительно масштабируем все данные
x_scaled = StandardScaler().fit_transform(x)

cv_scores = cross_val_score(model, x_scaled, y, cv=5)  # 5-фолдовая кросс-валидация

print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
