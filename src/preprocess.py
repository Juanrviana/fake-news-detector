import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Baixar os recursos necessários
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    #1. Converte o texto para letras minúsculas
    text = text.lower()

    #2. Remove números
    text = re.sub(r'\d+', '', text)

    #3. Remove pontuação e caracteres especiais
    text = re.sub(r'[^\w\s]', '', text)

    #4. tokeniza (quebra o texto em palavras)
    tokens = nltk.word_tokenize(text)

    #5. remove stopwords
    tokens = [word for word in tokens if word not in stop_words]

    #6. lematiza as palavras
    lematized = [lemmatizer.lemmatize(word) for word in tokens]

    # Retorna o texto limpo como string
    return ' '.join(lemmatizer)
