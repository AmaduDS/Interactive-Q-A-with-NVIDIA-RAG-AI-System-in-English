import streamlit as st
import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time

# Charger les variables d'environnement depuis le fichier .env
#from dotenv import load_dotenv
#load_dotenv()

# Récupérer la clé API NVIDIA pour LLaMA
#llama_api_key = os.getenv("NVIDIA_API_KEY")

# Remplacez l'accès aux clés API par st.secrets
llama_api_key = st.secrets["NVIDIA_API_KEY"]

# Ajouter une barre latérale pour les paramètres
st.sidebar.title("Settings")

# Paramètres de modèle
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.5)
top_p = st.sidebar.slider("Top_p", 0.0, 1.0, 1.0)

# Fonction de création des embeddings (Optimisée)
def vector_embedding_llama():
    if "vectors_en" not in st.session_state:
        st.session_state.embeddings = NVIDIAEmbeddings(api_key=llama_api_key)
        st.session_state.loader = PyPDFDirectoryLoader("./doc_en")  # Chemin vers le dossier de documents en anglais
        st.session_state.docs = st.session_state.loader.load()  # Chargement des documents

        if len(st.session_state.docs) == 0:
            st.error("Erreur : No document has been loaded. Check the file path.")
        else:
            # Diviser le document en chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            st.session_state.final_documents = text_splitter.split_documents(st.session_state.docs)

            if len(st.session_state.final_documents) == 0:
                st.error("Erreur : Impossible to divide documents into chunks.")
            else:
                try:
                    st.session_state.vectors_en = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
                    st.sidebar.success("FAISS Vector Store DB is ready with NVIDIA Embeddings (English).")
                except IndexError as e:
                    st.error(f"Error when creating vectors : {e}")

# Bouton pour faire l'embedding dans la barre latérale
if st.sidebar.button("Documents Embedding (English)"):
    vector_embedding_llama()

# Ajouter une icône à côté du titre
# Ajouter du CSS personnalisé pour ajuster la taille du titre
st.markdown("""
    <style>
    .custom-title {
        font-size: 24px;  /* Ajustez ici la taille de la police à votre convenance */
        font-weight: bold;
        color: #333333;  /* Couleur de la police */
        margin-bottom: 20px;  /* Espacement en dessous du titre */
    }
    </style>
    <h1 class="custom-title">📚 Canadian Citizenship Exam Preparation Guide: Interactive Q&A with NVIDIA RAG AI System</h1>
""", unsafe_allow_html=True)

# Définir le modèle en utilisant LLaMA pour l'anglais
try:
    llm = ChatNVIDIA(
        model="meta/llama3-70b-instruct",
        api_key=llama_api_key,
        base_url="https://integrate.api.nvidia.com/v1"
    )
    st.write("Modèle : LLama3-70b-instruct activated for english.")
except ValueError as e:
    st.error(f"Error when initializing the model : {e}")

# Créer un prompt template pour l'anglais
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

# Entrée de l'utilisateur pour la question
prompt1 = st.text_input("Enter your question (in English)")

# Si une question est posée
if prompt1:
    if "vectors_en" not in st.session_state:
        st.error("Please create document embeddings first using the 'Documents Embedding' button.")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors_en.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        st.write(f"Response time: {time.process_time() - start}")
        st.write(response['answer'])

        # Afficher les documents similaires
        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.write(f"Document {i + 1}: {doc.page_content[:500]}...")
                st.write("--------------------------------")
