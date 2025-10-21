import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
import gradio as gr

load_dotenv()

# LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0.7,
    top_p=0.9,
    top_k=40,
    google_api_key=os.getenv("GOOGLE_API_KEY")
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

def ask_question(question):
    """Soruyu RAG chain'e gönder ve cevabı döndür"""
    if not question or question.strip() == "":
        return "Lütfen bir soru girin."
    
    try:
        response = ""
        for chunk in rag_chain.stream(question):
            response += chunk
        return response
    except Exception as e:
        return f"Hata: {str(e)}"

# Gradio Arayüzü
with gr.Blocks(title="ChatDocs - AI Document Chat") as demo:
    gr.Markdown("# ChatDocs - AI ile Döküman Sohbeti")
    gr.Markdown("### 📖 Lilian Weng'in LLM Hallucination blog yazısı hakkında sorular sorun")
    
    with gr.Row():
        with gr.Column():
            question_input = gr.Textbox(
                label="Sorunuz",
                placeholder="Örnek: What is hallucination in LLMs?",
                lines=3
            )
            submit_btn = gr.Button("Gönder", variant="primary")
        
        with gr.Column():
            answer_output = gr.Textbox(
                label="Cevap",
                lines=10,
                show_copy_button=True
            )
    
    # Örnek sorular
    gr.Examples(
        examples=[
            "What is hallucination in LLMs?",
            "What are the main causes of hallucinations?",
            "How can we detect hallucinations?",
            "What is extrinsic hallucination?",
            "What are pre-training data issues?",
            "What is FActScore?",
            "What is Self-RAG?",
            "What is the capital of Turkey?"
        ],
        inputs=question_input
    )
    
    submit_btn.click(
        fn=ask_question,
        inputs=question_input,
        outputs=answer_output
    )
    
    question_input.submit(
        fn=ask_question,
        inputs=question_input,
        outputs=answer_output
    )

if __name__ == "__main__":
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860)