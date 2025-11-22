import joblib
import os
import sys

# Caminho do projeto
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)

from src.preprocess import preprocess_text

# Carregar modelo e vetorizador
model_path = os.path.join(BASE_DIR, "models/fake_news_model,pkl")
vectorizer_path = os.path.join(BASE_DIR, "models/vectorizer.pkl")

try:
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
except:
    print("ERRO: Modelo ou vetorizador não encontrado. Treino novamente com train.py ")
    sys.exit(1)

def prever(texto):
    texto_pre = preprocess_text(texto)
    vetor = vectorizer.transform([texto_pre])
    pred = model.predict(vetor)[0]
    prob = model.predict_proba(vetor)[0]

    return ("Fake", prob[0]) if pred == 0 else ("Real", prob[1])

def main():
    print("\n============================")
    print("   Fake NEWS DETECTOR")
    print("============================\n")

while True:
    texto = input("Digite uma notícia para classificar (ou ''exit' para sair):")

    if texto.lower() in ["exit", "sair", "quit"]:
        print("\nAté logo! ")
        break
    rotulo, confianca = prever(texto)

    print(f"\n Resultado: {rotulo}")
    print(f"Confiança: {confianca * 100:.2f}%")
    print("---------------------------\n")

if __name__ == "__main__":
    main()