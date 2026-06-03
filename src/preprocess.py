import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer  # Alterado para Stemmer em Português

# Adicione o path antes de carregar recursos
nltk.data.path.append(r"C:\Users\Juan\AppData\Roaming\nltk_data")

# Baixar os recursos necessários
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('rslp', quiet=True)  # Recurso necessário para o stemmer em PT-BR

# Inicialização
stemmer = RSLPStemmer()
# Alterado para carregar as stopwords em português
stop_words = set(stopwords.words('portuguese')) 

def preprocess_text(text):
    # Caso o texto venha nulo por alguma falha no dataset
    if not isinstance(text, str):
        return ""

    # 1. Minúsculas
    text = text.lower()

    # 2. Remove pontuação e números usando Regex simples
    text = re.sub(r'[^a-zA-ZáàâãéèêíïóôõöúçÑñÁÀÂÃÉÈÍÏÓÔÕÖÚÇ\s]', ' ', text)

    # 3. Tokenização
    tokens = nltk.word_tokenize(text)

    # 4. Remove stopwords e garante que são apenas letras
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]

    # 5. Stemming (Reduz as palavras ao radical em português)
    # Exemplo: "notícias" e "noticiou" viram "notic"
    tokens = [stemmer.stem(word) for word in tokens]

    # Retorna texto processado separado por espaços
    return ' '.join(tokens)