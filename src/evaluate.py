import os
import sys
import joblib

# Ajusta caminho para acessar src/
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from src.preprocess import preprocess_text


# ======================
#  Função de previsão
# ======================
def prever_texto(texto):
    # Carregar modelo e vetorizador
    model_path = os.path.join(BASE_DIR, "models", "fake_news_model.pkl")
    vect_path = os.path.join(BASE_DIR, "models", "vectorizer.pkl")

    if not os.path.exists(model_path) or not os.path.exists(vect_path):
        print("❌ ERRO: Modelo não encontrado! Treine usando train.py primeiro.")
        return None

    model = joblib.load(model_path)
    vectorizer = joblib.load(vect_path)

    # Pré-processar texto
    texto_processado = preprocess_text(texto)

    # Vetorizar
    texto_vetorizado = vectorizer.transform([texto_processado])

    # Previsão
    pred = model.predict(texto_vetorizado)[0]
    prob = model.predict_proba(texto_vetorizado)[0]

    # Resultado
    classe = "Real" if pred == 1 else "Fake"
    confianca = round(max(prob) * 100, 2)

    return classe, confianca


# ======================
#  Execução direta
# ======================
if __name__ == "__main__":
    print("=== CLASSIFICADOR DE FAKE NEWS ===")

    texto = input("\nDigite o texto da notícia para analisar:\n> ")

    resultado = prever_texto(texto)

    if resultado:
        classe, confianca = resultado
        print(f"\n🔎 Classificação: **{classe}**")
        print(f"📊 Confiança: **{confianca}%**")
