import streamlit as st
import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv()

# Sayfa Yapılandırması
st.set_page_config(
    page_title="ChatDocs - AI Document Chat",
    page_icon="📚",
    layout="wide"
)

# Başlık
st.title("📚 ChatDocs - AI ile Döküman Sohbeti")
st.markdown("### 📖 Lilian Weng'in LLM Hallucination blog yazısı hakkında sorular sorun")

# Sidebar
with st.sidebar:
    st.header("ℹ️ Hakkında")
    st.info("""
    **ChatDocs**, RAG (Retrieval Augmented Generation) mimarisi kullanarak 
    web içeriklerinden akıllı soru-cevap yapabilen yapay zeka destekli bir uygulamadır.
    """)
    
    st.header("Örnek Sorular")
    example_questions = [
        "What is hallucination in LLMs?",
        "What are the main causes of hallucinations?",
        "How can we detect hallucinations?",
        "What is extrinsic hallucination?",
        "What are pre-training data issues?",
        "What is FActScore?",
        "What is Self-RAG?",
        "What is the capital of Turkey?"
    ]
    
    for i, question in enumerate(example_questions):
        if st.button(question, key=f"example_{i}"):
            st.session_state.selected_question = question

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'selected_question' not in st.session_state:
    st.session_state.selected_question = None

# Cache fonksiyonu - RAG bileşenlerini bir kere yükle
@st.cache_resource
def load_rag_chain():
    # API Key - Streamlit secrets veya .env'den al
    api_key = st.secrets.get("GOOGLE_API_KEY") if "GOOGLE_API_KEY" in st.secrets else os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        st.error("GOOGLE_API_KEY bulunamadı! Lütfen .env dosyanızı kontrol edin.")
        st.stop()
    
    # LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        google_api_key=api_key
    )

    # Load chunk and index the contents of the blog
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2024-07-07-hallucination/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # HuggingFace Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings
    )

    # Retrieve and generate using the relevant snippets of the blog
    retriever = vectorstore.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# RAG chain'i yükle
with st.spinner("AI modeli yükleniyor..."):
    rag_chain = load_rag_chain()

# Chat mesajlarını göster
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Örnek sorudan gelen input
if st.session_state.selected_question:
    prompt = st.session_state.selected_question
    st.session_state.selected_question = None
else:
    # Chat input
    prompt = st.chat_input("Sorunuzu buraya yazın...")

if prompt:
    # Kullanıcı mesajını ekle
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # AI cevabını al
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Streaming response
        with st.spinner("Düşünüyorum..."):
            try:
                for chunk in rag_chain.stream(prompt):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            except Exception as e:
                error_message = f"Hata: {str(e)}"
                message_placeholder.markdown(error_message)
                full_response = error_message
    
    # Assistant mesajını kaydet
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Powered by Google Gemini 2.0 Flash • LangChain • Streamlit</p>
    <p>📖 Source: <a href='https://lilianweng.github.io/posts/2024-07-07-hallucination/' target='_blank'>LLM Hallucination by Lilian Weng</a></p>
</div>
""", unsafe_allow_html=True)
