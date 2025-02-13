import streamlit as st
import os
import logging
import chromadb
import requests
import sqlite3
import bcrypt
from bs4 import BeautifulSoup
import PyPDF2
from duckduckgo_search import DDGS
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM

def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT UNIQUE, password TEXT)''')
    conn.commit()
    conn.close()

def register_user(username, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False

def authenticate_user(username, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username = ?", (username,))
    user = c.fetchone()
    conn.close()
    if user and bcrypt.checkpw(password.encode(), user[0].encode()):
        return True
    return False

def process_uploaded_pdf(file, filename):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text_content = []
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                text_content.append(text)
        full_text = "\n".join(text_content)
        if full_text:
            add_or_update_documents([{"id": f"pdf-{filename}", "text": full_text}])
    except Exception as e:
        st.error(f"Error processing PDF: {e}")

def scrape_with_duckduckgo(query, max_results=3):
    results = []
    
    try:
        with DDGS() as ddgs:
            search_results = ddgs.text(query, max_results=max_results)
            
            for result in search_results:
                url = result.get("href")
                snippet = result.get("body", "")

                if not url:
                    continue

                try:
                    response = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
                    if response.status_code != 200:
                        continue
                    soup = BeautifulSoup(response.text, "html.parser")

                    paragraphs = soup.find_all("p")
                    page_text = "\n".join(p.get_text() for p in paragraphs)

                    if len(page_text) > 500:
                        page_text = page_text[:500] + "..."

                    images = [img["src"] for img in soup.find_all("img", src=True) if img["src"].startswith("http")]

                    doc_text = f"Snippet: {snippet}\n\nFull Page Text:\n{page_text}"

                    results.append({"id": f"web-{url}", "text": doc_text, "images": images})

                except requests.exceptions.RequestException as e:
                    logging.warning(f"Error loading {url}: {e}")

    except Exception as e:
        logging.error(f"DuckDuckGo search error: {e}")
        return []

    return results

def process_uploaded_txt(file, filename):
    try:
        content = file.read().decode("utf-8").strip()
        if content:
            add_or_update_documents([{"id": f"txt-{filename}", "text": content}])
        else:
            st.error("The file is empty.")
    except Exception as e:
        st.error(f"Error processing TXT: {e}")

def scrape_and_display_images(query, max_results=3):
    results = scrape_with_duckduckgo(query, max_results)
    if results:
        for result in results:
            st.write(result["text"])
            if "images" in result and result["images"]:
                for img_url in result["images"][:5]:
                    st.image(img_url, caption="Relevant image", width=300)
    else:
        st.write("No results found.")

CHROMA_DB_DIR = "chroma_db"
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
collection_name = "knowledge_base"
collection = chroma_client.get_or_create_collection(
    name=collection_name, metadata={"description": "Knowledge base for RAG"}
)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def add_or_update_documents(documents):
    if not documents:
        st.error("No data to add.")
        return
    doc_texts = [doc["text"] for doc in documents]
    doc_ids = [doc["id"] for doc in documents]
    embeddings = embedding_model.encode(doc_texts)
    try:
        collection.add(documents=doc_texts, ids=doc_ids, embeddings=embeddings)
        st.success(f"Added {len(documents)} documents to the knowledge base.")
    except Exception as e:
        st.error(f"Error adding documents: {e}")

def view_knowledge_base():
    results = collection.get()
    docs = results.get("documents", [])
    ids = results.get("ids", [])
    st.subheader("📂 All documents in the knowledge base:")
    if docs:
        for idx, (doc_id, doc) in enumerate(zip(ids, docs), start=1):
            st.write(f"**{idx}.** `{doc_id}`\n{doc[:500]}...")
            st.markdown("---")
    else:
        st.write("❌ There are no documents in the database yet.")

def query_knowledge_base(query_text, n_results=3):
    query_embedding = embedding_model.encode([query_text])
    results = collection.query(query_embeddings=query_embedding, n_results=n_results)
    docs = results.get("documents", [])
    return docs

def query_ollama(prompt):
    llm = OllamaLLM(model="llama3.2")
    return llm.invoke(prompt)

def rag_pipeline(query_text):
    retrieved_docs = query_knowledge_base(query_text)
    flattened_docs = []
    for doc in retrieved_docs:
        if isinstance(doc, list):
            flattened_docs.extend(doc)
        else:
            flattened_docs.append(doc)
    context = " ".join(flattened_docs) if flattened_docs else "No relevant documents found."
    prompt = f"Context: {context}\n\nQuestion: {query_text}\nAnswer:"
    response = query_ollama(prompt)
    return response

def main():
    st.title("📚 Knowledge Base with AI-answers (RAG) and Authentication")
    init_db()
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if not st.session_state.authenticated:
        st.sidebar.header("🔑 Authentication")
        choice = st.sidebar.radio("Select option", ["Login", "Register"])
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        if choice == "Login":
            if st.sidebar.button("Login"):
                if authenticate_user(username, password):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.success("Logged in successfully!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
        if choice == "Register":
            if st.sidebar.button("Register"):
                if register_user(username, password):
                    st.success("User registered successfully! Please log in.")
                else:
                    st.error("Username already exists")
        return
    st.sidebar.write(f"Logged in as: {st.session_state.username}")
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()
    tab1, tab2, tab3, tab4 = st.tabs(["📂 Knowledge Base", "📥 Upload Documents", "🔍 Web Search", "🤖 AI-answer"])
    with tab1:
        st.header("📂 Knowledge Base")
        view_knowledge_base()
    with tab2:
        st.header("📥 Upload Documents")
        uploaded_pdf = st.file_uploader("📄 Upload PDF file", type=["pdf"])
        if uploaded_pdf:
            process_uploaded_pdf(uploaded_pdf, uploaded_pdf.name)
        uploaded_txt = st.file_uploader("📄 Upload TXT file", type=["txt"])
        if uploaded_txt:
            process_uploaded_txt(uploaded_txt, uploaded_txt.name)
    with tab3:
        st.header("🔍 Web Search")
        search_query = st.text_input("Enter query:")
        if st.button("🔎 Search the Internet"):
            if search_query.strip():
                scrape_and_display_images(search_query)
    with tab4:
        st.header("🤖 AI-answer (RAG)")
        user_question = st.text_input("Enter your question:")
        if st.button("💬 Get Answer"):
            response = rag_pipeline(user_question)
            st.subheader("📝 AI Answer:")
            st.write(response)

if __name__ == "__main__":
    main()
