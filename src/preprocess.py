import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Adicione o path antes de carregar recursos
nltk.data.path.append(r"C:\Users\Juan\AppData\Roaming\nltk_data")

# Baixar os recursos necessários
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)

# Inicialização
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # 1. Minúsculas
    text = text.lower()

    # 2. Remove números
    text = re.sub(r'\d+', '', text)

    # 3. Remove pontuação
    text = re.sub(r'[^\w\s]', ' ', text)

    # 4. Tokenização
    tokens = nltk.word_tokenize(text)

    # 5. Remove stopwords e tokens inválidos
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]

    # 6. Lemmatização
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Retorna texto processado
    return ' '.join(tokens)
