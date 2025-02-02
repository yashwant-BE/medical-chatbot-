from flask import Flask, render_template, request, redirect, url_for, flash
import google.generativeai as genai
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.document_loaders import PyMuPDFLoader  # Use PyMuPDFLoader instead of PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import logging
import pickle

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = 'supersecretkey'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Gemini model
my_api_key_gemini = "AIzaSyCeN64TLXot9zd1R1vqQAFY1jY2AvkVh5U"  # Replace with your Gemini API key
genai.configure(api_key=my_api_key_gemini)
model = genai.GenerativeModel('gemini-2.0-flash-exp')

# Initialize vector store
vector_store = None
if os.path.exists('vector_store.pkl'):
    with open('vector_store.pkl', 'rb') as f:
        vector_store = pickle.load(f)

# Error handler for 404
@app.errorhandler(404)
def page_not_found(e):
    return redirect(url_for('index'))

# Main page route
@app.route('/')
def index():
    return render_template('index.html')

# File upload route
@app.route('/upload', methods=['POST'])
def upload():
    global vector_store
    try:
        if 'pdf_files' not in request.files:
            flash("No file part")
            return redirect(url_for('index'))

        files = request.files.getlist('pdf_files')
        documents = []

        for file in files:
            if file.filename == '':
                flash("No selected file")
                return redirect(url_for('index'))

            # Save the file to the uploads folder
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            print(f"File saved: {file_path}")

            try:
                # Load the PDF file using PyMuPDFLoader
                pdf_loader = PyMuPDFLoader(file_path)
                documents.extend(pdf_loader.load())
                print(f"PDF loaded: {file_path}")
            except Exception as e:
                logger.error(f"Error loading PDF {file.filename}: {e}")
                flash(f"Error loading PDF {file.filename}. Please ensure it is a valid PDF file.")
                return redirect(url_for('index'))

        # Create embeddings using HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings()
        print("Embeddings created")

        if vector_store is None:
            # Create a new vector store if it doesn't exist
            vector_store = FAISS.from_documents(documents, embeddings)
            print("New vector store created")
        else:
            # Add new documents to the existing vector store
            vector_store.add_documents(documents)
            print("Documents added to existing vector store")

        # Save the updated vector store
        with open('vector_store.pkl', 'wb') as f:
            pickle.dump(vector_store, f)
        print("Vector store saved")

        flash("PDFs uploaded and processed successfully. The knowledge base is ready.")
        return redirect(url_for('index'))
    except Exception as e:
        logger.error("An error occurred while processing the PDFs: %s", e)
        flash("An error occurred while processing the PDFs.")
        return redirect(url_for('index'))

# Question answering route
@app.route('/ask', methods=['POST'])
def ask():
    global vector_store
    if vector_store is None:
        return "Knowledge base is not ready. Please upload PDFs first."

    question = request.form['prompt']
    # Retrieve relevant documents based on the question
    relevant_docs = vector_store.similarity_search(question)
    context = " ".join([doc.page_content for doc in relevant_docs])
    custom_prompt = f"You are the best doctor. Only provide medical-related answers. Context: {context} Question: {question}"

    response = model.generate_content(custom_prompt)

    if response.text:
        return response.text
    else:
        return "Sorry, but I think Gemini didn't want to answer that!"

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
