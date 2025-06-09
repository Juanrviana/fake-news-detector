import pandas as pd
import sys
import os
import nltk
nltk.download('punkt')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from src.preprocess import preprocess_text

# Carrega os datasets
fake = pd.read_csv('data/raw/Fake.csv')
real = pd.read_csv('data/raw/True.csv')

#Adiciona o rótulo
fake['label'] = 0 # falso
real['label'] = 1 # verdadeiro

#Junta os dois
df = pd.concat([fake, real]).sample(frac=1).reset_index(drop=True)

#Usa só a coluna 'text' e 'label'
texts = df['text'].apply(preprocess_text) # pré-processa
labels = df['label']

#Separa treino/tete
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

#Tramsforma texto em vetores numéricos com TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Treina o modelo
model = LogisticRegression()
model.fit(X_train_vec, y_train)

#Avalia
preds = model.predict(X_test_vec)
print(classification_report(y_test, preds))