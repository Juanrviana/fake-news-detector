import joblib
import os
import sys

# Ajusta caminho raiz
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from src.preprocess import preprocess_text

# Carrega modelo e vetorizador
model = joblib.load(os.path.join(BASE_DIR, "models/fake_news_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "models/vectorizer.pkl"))

def prever_texto(texto: str):
    # 1. Pré-processa o texto
    texto_processado = preprocess_text(texto)

    # 2. Vetoriza
    texto_vetorizado = vectorizer.transform([texto_processado])

    # 3. Faz a previsão
    pred = model.predict(texto_vetorizado)

    # 4. Retorna resultado
    return "Fake" if pred[0] == 0 else "Real"

if __name__ == "__main__":
    entrada = input("Digite a notícia para classificar: ")
    resultado = prever_texto(entrada)
    print(f"Classificação: {resultado}")
