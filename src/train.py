import pandas as pd
import os
import sys
import joblib

# Caminho raiz do projeto
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from src.preprocess import preprocess_text

# ======== 1. Carregar dados ========
fake_path = os.path.join(BASE_DIR, "data/raw/Fake.csv")
real_path = os.path.join(BASE_DIR, "data/raw/True.csv")

fake = pd.read_csv(fake_path)
real = pd.read_csv(real_path)

# Rótulos
fake["label"] = 0
real["label"] = 1

# Unir datasets
df = pd.concat([fake, real]).sample(frac=1).reset_index(drop=True)

# ======== 2. Dividir treino/teste antes do preprocess ========
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

# ======== 3. Pré-processar ========
X_train_clean = X_train.apply(preprocess_text)
X_test_clean = X_test.apply(preprocess_text)

# ======== 4. Vetorização ========
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train_clean)
X_test_vec = vectorizer.transform(X_test_clean)

# ======== 5. Modelo ========
model = LogisticRegression(max_iter=200)
model.fit(X_train_vec, y_train)

# ======== 6. Avaliação ========
preds = model.predict(X_test_vec)

print("===== RELATÓRIO DE CLASSIFICAÇÃO =====")
print(classification_report(y_test, preds))

print("===== MATRIZ DE CONFUSÃO =====")
print(confusion_matrix(y_test, preds))

# ======== 7. Salvar modelo e vetorizador ========
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
joblib.dump(model, os.path.join(BASE_DIR, "models/fake_news_model.pkl"))
joblib.dump(vectorizer, os.path.join(BASE_DIR, "models/vectorizer.pkl"))

print("\nModelo e vetorizador salvos com sucesso!")
