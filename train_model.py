# train_model.py

# Mengimpor library yang dibutuhkan
import pandas as pd
import numpy as np
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Membaca dan membersihkan dataset
data = pd.read_csv('D:/shaff/kuliah/#semester4/machine learning/IMDB Dataset.csv')
data = data.drop_duplicates()  # Menghapus data duplikat
data = data.dropna()  # Menghapus baris dengan data yang tidak lengkap
data['review'] = data['review'].str.lower()  # Mengonversi teks ulasan menjadi huruf kecil

# Membagi data menjadi fitur (X) dan label (y)
X = data['review']
y = data['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

# Membagi data menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Mengonversi teks menjadi fitur TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

from sklearn.naive_bayes import MultinomialNB

#Pelatihan Model Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
# Memprediksi pada data uji
y_pred = nb_model.predict(X_test_tfidf)
# Evaluasi model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Melatih model Logistic Regression
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)
# Memprediksi pada data uji
y_pred = model.predict(X_test_tfidf)
# Evaluasi model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Grid Search untuk optimasi hyperparameter
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1, 1, 10], 'solver': ['newton-cg', 'lbfgs', 'liblinear']}
grid = GridSearchCV(LogisticRegression(), param_grid, refit=True, verbose=2)
grid.fit(X_train_tfidf, y_train)

# Hasil terbaik
print("hasil terbaik")
print(grid.best_params_)
print(grid.best_estimator_)

# Simpan model dan vectorizer ke dalam file 'models/sentiment_model.pkl' dan 'models/vectorizer.pkl'
joblib.dump(model, 'models/sentiment_model.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')
