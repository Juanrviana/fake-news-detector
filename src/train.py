import os
import joblib  # Necessário para salvar o modelo e o vetorizador (.pkl)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Importa a função corrigida do teu preprocess.py
from preprocess import preprocess_text 

def load_and_prepare_data():
    print("[1/5] A carregar os datasets...")
    
    # Caminhos baseados na árvore de pastas do teu projeto
    fake_path = os.path.join("data", "raw", "Fake.csv")
    true_path = os.path.join("data", "raw", "True.csv")
    
    # Validação rápida de caminhos
    if not os.path.exists(fake_path) or not os.path.exists(true_path):
        raise FileNotFoundError("Garante que estás a correr o script a partir da raiz do projeto e que Fake.csv/True.csv estão em data/raw/")

    # Lendo os CSVs
    df_fake = pd.read_csv(fake_path)
    df_true = pd.read_csv(true_path)
    
    # Criando a coluna label (0 para verdadeiro, 1 para fake)
    df_true['label'] = 0
    df_fake['label'] = 1
    
    # Juntando e misturando as duas bases
    df_total = pd.concat([df_true, df_fake], ignore_index=True)
    df_total = df_total.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df_total

def main():
    # 1. Carregar os dados
    df = load_and_prepare_data()
    
    # Define a coluna que contém o texto da notícia. Altere se no CSV for outro nome (ex: 'text' ou 'title')
    coluna_texto = 'text' 
    
    # 2. Pré-processamento do Texto
    print("[2/5] A iniciar o pré-processamento (isto pode demorar alguns minutos)...")
    df['text_cleaned'] = df[coluna_texto].apply(preprocess_text)
    
    # Remover eventuais linhas que ficaram vazias após a limpeza
    df = df[df['text_cleaned'] != ""].reset_index(drop=True)
    
    # 3. Divisão de Treino e Teste (80% treino, 20% teste)
    print("[3/5] A dividir os dados em Treino e Teste...")
    X = df['text_cleaned']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 4. Vetorização com TF-IDF
    print("[4/5] A transformar texto em números com TF-IDF...")
    # max_features impede o modelo de ficar gigantesco e foca nas palavras mais importantes
    vectorizer = TfidfVectorizer(max_features=5000) 
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # 5. Treino do Modelo (Regressão Logística é excelente para classificação de texto)
    print("[5/5] A treinar o modelo de Regressão Logística...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)
    
    # Avaliação do Modelo
    y_pred = model.predict(X_test_vec)
    print("\n=== RESULTADOS DO MODELO ===")
    print(f"Acurácia Geral: {accuracy_score(y_test, y_pred):.4f}")
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred, target_names=['Verdadeira (0)', 'Fake (1)']))
    
    # 6. Salvar Artefactos nas pastas corretas
    print("\nA gravar o modelo e o vetorizador...")
    os.makedirs("models", exist_ok=True) # Garante que a pasta existe
    
    joblib.dump(vectorizer, os.path.join("models", "vectorizer.pkl"))
    joblib.dump(model, os.path.join("models", "fake_news_model.pkl"))
    
    print("Sucesso! Os ficheiros 'vectorizer.pkl' e 'fake_news_model.pkl' foram gerados na pasta 'models'.")

if __name__ == "__main__":
    main()