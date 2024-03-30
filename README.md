# Задание № 1


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Загрузим данные
train_data = pd.read_csv('C:\\Users\\Apple\\Downloads\\train.csv')
test_data = pd.read_csv('c:\\Users\\Apple\\Downloads\\test.csv')

# Предобработка данных
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train = tfidf_vectorizer.fit_transform(train_data['text'])
y_train = train_data['sentiment']

X_test = tfidf_vectorizer.transform(test_data['text'])

# Разделим данные на обучающий и валидационный наборы
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Обучение модели
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Предсказание на валидационном наборе
y_pred = model.predict(X_val)
test_predictions = model.predict(X_test)

# Добавление столбца 'id' на основе индексов test_data
test_data['id'] = test_data.index

# Создание DataFrame для submission
submission = pd.DataFrame({'id': test_data['id'], 'sentiment': test_predictions})
# Сохранение предсказаний в файл
submission = pd.DataFrame({'id': test_data['id'], 'sentiment': test_predictions})
submission.to_csv('c:\\Users\\Apple\\Desktop\\Данные с Windows\\Users\\User\\Desktop\\мои документы\\Задание 1.3.csv', index=False)
print('Предсказания сохранены в файл.')
