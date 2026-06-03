import os
import joblib
# Importa o mesmo pré-processamento que usamos no treino
from preprocess import preprocess_text 

def predict_news(text):
    # 1. Carregar o vetorizador e o modelo salvos
    vectorizer_path = os.path.join("models", "vectorizer.pkl")
    model_path = os.path.join("models", "fake_news_model.pkl")
    
    if not os.path.exists(vectorizer_path) or not os.path.exists(model_path):
        raise FileNotFoundError("Modelo ou Vetorizador não encontrados na pasta 'models'. Rode o train.py primeiro.")
        
    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path)
    
    # 2. Aplicar a limpeza no texto inserido
    text_cleaned = preprocess_text(text)
    
    # 3. Vetorizar o texto (transformar em números)
    text_vec = vectorizer.transform([text_cleaned])
    
    # 4. Fazer a previsão e calcular a probabilidade
    prediction = model.predict(text_vec)[0]
    probabilities = model.predict_proba(text_vec)[0]
    
    # 5. Retornar os resultados formatados
    resultado = "FAKE NEWS" if prediction == 1 else "NOTÍCIA VERDADEIRA"
    confianca = probabilities[prediction] * 100
    
    return resultado, confianca

if __name__ == "__main__":
    print("=== DETECTOR DE FAKE NEWS PRONTO ===")
    print("Digite a notícia abaixo para analisar (ou 'sair' para encerrar):")
    
    while True:
        noticia = input("\nInsira o texto da notícia: ")
        if noticia.lower() == 'sair':
            break
            
        if not noticia.strip():
            print("Por favor, digite algum texto.")
            continue
            
        resultado, confianca = predict_news(noticia)
        print(f"\nResultado: O modelo identificou como [{resultado}]")
        print(f"Confiança da IA: {confianca:.2f}%")