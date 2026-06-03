import os
import sys
import streamlit as st
import joblib

# Pega o caminho absoluto do diretório onde o app.py está e sobe um nível para achar a raiz
diretorio_atual = os.path.dirname(os.path.abspath(__file__))
raiz_do_projeto = os.path.abspath(os.path.join(diretorio_atual, ".."))

# Adiciona a raiz e a pasta src no caminho de busca do Python
if raiz_do_projeto not in sys.path:
    sys.path.append(raiz_do_projeto)
    sys.path.append(os.path.join(raiz_do_projeto, "src"))

# Agora o Python vai encontrar o preprocess_text sem problemas!
from preprocess import preprocess_text

# Garante que o Python encontre o pacote 'src' para importar o preprocess_text
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from preprocess import preprocess_text

# Configuração da página do navegador
st.set_page_config(
    page_title="Detector de Fake News",
    page_icon="📰",
    layout="centered"
)

# Carrega os modelos salvos
@st.cache_resource
def carregar_modelo():
    vectorizer_path = os.path.join("models", "vectorizer.pkl")
    model_path = os.path.join("models", "fake_news_model.pkl")
    
    if not os.path.exists(vectorizer_path) or not os.path.exists(model_path):
        return None, None
        
    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path)
    return vectorizer, model

vectorizer, model = carregar_modelo()

# Interface Visual
st.title("📰 Detector Inteligente de Fake News")
st.markdown("Cole o texto de uma notícia abaixo para verificar a probabilidade de ela ser falsa ou verdadeira.")

# Área de texto para o usuário colar a notícia
noticia_input = st.text_area("Texto da Notícia:", height=250, placeholder="Cole o conteúdo completo da notícia aqui...")

if st.button("Analisar Notícia", type="primary"):
    if not noticia_input.strip():
        st.warning("⚠️ Por favor, insira algum texto para analisar.")
    elif vectorizer is None or model is None:
        st.error("❌ Erro: O modelo e o vetorizador não foram encontrados na pasta 'models'. Execute o 'train.py' primeiro.")
    else:
        with st.spinner("Analisando os padrões do texto..."):
            # 1. Pré-processamento
            texto_limpo = preprocess_text(noticia_input)
            
            # 2. Vetorização e Previsão
            texto_vec = vectorizer.transform([texto_limpo])
            predicao = model.predict(texto_vec)[0]
            probabilidades = model.predict_proba(texto_vec)[0]
            
            confianca = probabilidades[predicao] * 100
            
            st.write("---")
            st.subheader("Resultado da Análise:")
            
            # 3. Exibição do Resultado baseado na predição (1=Fake, 0=True)
            if predicao == 1:
                st.error(f"🚨 **ALERTA: Esta notícia tem grandes chances de ser FAKE NEWS!**")
                st.metric(label="Confiança da IA", value=f"{confianca:.2f}%")
                st.progress(int(confianca))
            else:
                st.success(f"✅ **CONFIRMADO: Esta notícia parece ser VERDADEIRA.**")
                st.metric(label="Confiança da IA", value=f"{confianca:.2f}%")
                st.progress(int(confianca))

st.write("---")
st.caption("Desenvolvido como projeto de detecção automática de desinformação utilizando NLP e Machine Learning.")