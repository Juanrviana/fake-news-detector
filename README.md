# 🕵️‍♂️ Fake News Detector (NLP & Machine Learning)

Aplicação desenvolvida em **Python** para classificação automática e verificação de veracidade de notícias em português, utilizando técnicas de **Processamento de Linguagem Natural (NLP)** e **Aprendizado de Máquina**.

---

## 📌 Visão Geral do Projeto

O objetivo deste projeto é identificar e classificar notícias falsas (fake news) a partir da análise do texto do artigo ou manchete. A solução processa o texto bruto, extrai características linguísticas relevantes e aplica modelos de classificação treinados para identificar padrões associados à desinformação.

---

## 🛠️ Tecnologias e Bibliotecas Utilizadas

- **Linguagem:** Python 3
- **Processamento de Linguagem Natural (NLP):** NLTK (Natural Language Toolkit)
- **Interface Web / Dashboard:** Streamlit
- **Manipulação de Dados:** Pandas / NumPy
- **Machine Learning:** Scikit-learn
- **Controle de Versão:** Git & GitHub

---

## ⚙️ Principais Funcionalidades

- **Pré-processamento de Texto:** Tokenização, remoção de *stopwords* e normalização de caracteres.
- **Extração de Características:** Vetorização e análise de frequência de termos (TF-IDF / Bag of Words).
- **Classificação:** Predição de veracidade com base no modelo preditivo treinado.
- **Interface Interativa:** Dashboard simples e responsivo em Streamlit para inserção de textos e exibição do resultado em tempo real.

---

## 📂 Estrutura do Repositório

```text
fake-news-detector/
├── data/                  # Conjunto de dados (datasets de notícias)
├── models/                # Modelos treinados e vetorizadores salvos (.pkl)
├── app.py                 # Aplicação web com interface Streamlit
├── preprocessor.py        # Módulo de limpeza e pré-processamento de texto
├── requirements.txt       # Dependências e bibliotecas do projeto
└── README.md              # Documentação do projeto

🚀 Como Executar o Projeto Localmente
Clone o repositório:
Bash
git clone [https://github.com/Juanrviana/fake-news-detector.git](https://github.com/Juanrviana/fake-news-detector.git)
cd fake-news-detector

Crie e ative um ambiente virtual (recomendado):
Bash
python -m venv venv
# No Linux/Mac:
source venv/bin/activate
# No Windows:
venv\Scripts\activate

Instale as dependências:
Bash
pip install -r requirements.txt

Execute a aplicação Streamlit:
Bash
streamlit run app.py

