# Задание № 1
В этом репозитории представлен код первого задания и его описание. Программа была обучена на файле "train", протетсирована на файле "test' и результат был выведен в "Задание 1.3". Общая идея решения заключается в использовании машинного обучения для классификации текстовых данных.

import pandas as pd - эта строка импортирует библиотеку pandas, которая предоставляет мощные инструменты для анализа данных, включая работу с таблицами данных (DataFrame).

from sklearn.feature_extraction.text import TfidfVectorizer - этот импорт загружает TfidfVectorizer из библиотеки scikit-learn, который используется для преобразования текстовых данных в числовые признаки с помощью метода TF-IDF

from sklearn.model_selection import train_test_split - эта строка импортирует функцию train_test_split из scikit-learn, которая используется для разделения данных на обучающий и тестовый наборы.

from sklearn.linear_model import LogisticRegression - этот импорт загружает модель логистической регрессии из scikit-learn, которая является одним из методов классификации.

from sklearn.metrics import accuracy_score, classification_report - этот импорт загружает функции accuracy_score и classification_report из scikit-learn, которые используются для оценки качества классификации.

Таким образом, эти библиотеки используются для создания и оценки модели машинного обучения для классификации текстовых данных.
