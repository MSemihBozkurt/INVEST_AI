import os
import subprocess
import platform
import google.generativeai as genai
from RAG.document_loader import load_documents_from_folder
from RAG.embedder import get_embedding_function
from RAG.vector_store import create_chroma_client, add_document_to_collection, retrieveDocs, summarize_collection, clear_collection, search_exact_term
from dotenv import load_dotenv
load_dotenv()

def build_chatBot(system_instruction):
    """Initialize the Gemini chat model."""
    try:
        model = genai.GenerativeModel('gemini-2.0-flash', system_instruction=system_instruction)
        return model.start_chat(history=[])
    except Exception as e:
        print(f"Error initializing Gemini model: {e}")
        raise

def generate_LLM_answer(prompt, context, chat):
    """Generate a response using the Gemini LLM."""
    try:
        response = chat.send_message(prompt + context)
        return response.text
    except Exception as e:
        print(f"Error generating LLM answer: {e}")
        return "Error: Unable to generate response."

def open_pdf(file_path):
    """Open PDF file using the default system application."""
    try:
        if platform.system() == 'Darwin':  # macOS
            subprocess.call(('open', file_path))
        elif platform.system() == 'Windows':  # Windows
            os.startfile(file_path)
        else:  # Linux
            subprocess.call(('xdg-open', file_path))
        print(f"Opening PDF: {file_path}")
    except Exception as e:
        print(f"Error opening PDF: {e}")

def get_source_documents(retrieved_results):
    """Extract unique source documents from retrieved results."""
    if not retrieved_results or 'metadatas' not in retrieved_results:
        return []
    
    sources = []
    seen_docs = set()
    
    for metadata in retrieved_results['metadatas'][0]:
        doc_name = metadata.get('document', 'Unknown')
        if doc_name not in seen_docs:
            seen_docs.add(doc_name)
            sources.append(doc_name)
    
    return sources

def display_citations(sources, document_folder):
    """Display citations with clickable links to open PDFs."""
    if not sources:
        return
    
    print("\n" + "="*50)
    print("SOURCES:")
    print("="*50)
    
    for i, source in enumerate(sources, 1):
        print(f"{i}. {source}")
        
        # Check if PDF exists in the document folder
        pdf_path = os.path.join(document_folder, source)
        if os.path.exists(pdf_path):
            while True:
                user_input = input(f"   Press 'o' to open this PDF, or Enter to continue: ").strip().lower()
                if user_input == 'o':
                    open_pdf(pdf_path)
                    break
                elif user_input == '':
                    break
                else:
                    print("   Invalid input. Press 'o' to open or Enter to continue.")
        else:
            print(f"   (PDF file not found at: {pdf_path})")
    
    print("="*50)

def generateAnswer(RAG_LLM, chroma_collection, query, document_folder, n_results=10, only_response=True):
    """Generate an answer using the enhanced RAG pipeline with citation support."""
    try:
        # First, try exact term search
        exact_matches = search_exact_term(chroma_collection, query)
        if exact_matches:
            print("------- Exact Term Matches Found -------")
            for i, match in enumerate(exact_matches[:3]):  # Show top 3 exact matches
                print(f"Exact Match {i+1}: {match['document'][:200]}...")
        
        # Then do semantic search
        retrieved_results = retrieveDocs(chroma_collection, query, n_results, return_only_docs=False)
        retrieved_documents = retrieved_results['documents'][0] if retrieved_results else []
        
        # Combine exact matches with semantic search results
        all_context = []
        if exact_matches:
            all_context.extend([match['document'] for match in exact_matches[:3]])
        all_context.extend(retrieved_documents[:7])  # Take top 7 from semantic search
        
        # Remove duplicates while preserving order
        seen = set()
        unique_context = []
        for doc in all_context:
            if doc not in seen:
                seen.add(doc)
                unique_context.append(doc)
        
        prompt = f"QUESTION: {query}"
        context = f"\nEXCERPTS: {chr(10).join(unique_context)}"
        
        if not only_response:
            print("------- Retrieved Documents -------")
            for i, doc in enumerate(unique_context):
                print(f"Document {i+1}: {doc[:150]}...")
            print("------- RAG Answer -------")
        
        output = generate_LLM_answer(prompt, context, RAG_LLM)
        
        # Get and display source documents
        sources = get_source_documents(retrieved_results)
        
        print(output)
        display_citations(sources, document_folder)
        
        return output
    except Exception as e:
        print(f"Error in RAG pipeline: {e}")
        return "Error: Unable to process query."

def main():
    """Main function to set up and run the enhanced RAG pipeline."""
    # Configuration
    collection_name = "Papers"
    sentence_transformer_model = "distiluse-base-multilingual-cased-v1"
    chromaDB_path = os.path.join(".", "chromadb_data")
    document_folder = os.path.join(".", "sample_data")
    api_key = os.getenv('GEMINI_API_KEY')

    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        return

    # Initialize Gemini with improved system prompt
    genai.configure(api_key=api_key)
    system_prompt = """You are a financial and legal terminology expert assistant. 
    Answer questions based on the provided context from financial glossaries and documents.
    
    Instructions:
    1. If you find a direct definition in the context, provide it clearly
    2. If the exact term isn't defined but related concepts are present, explain what you can infer
    3. If you find related terms that might help understand the concept, mention them
    4. If the information is truly not available, say so clearly
    5. Always be precise and cite specific information from the context when available
    
    Focus on providing helpful, accurate information from the financial domain."""
    
    RAG_LLM = build_chatBot(system_prompt)

    # Initialize ChromaDB and embedding function
    embedding_function = get_embedding_function(sentence_transformer_model)
    chroma_client, chroma_collection = create_chroma_client(collection_name, embedding_function, chromaDB_path)

    # Uncomment the next line if you want to rebuild the collection from scratch
    clear_collection(chroma_collection)

    # Load and process documents from sample_data folder
    if not os.path.exists(document_folder):
        print(f"Error: {document_folder} does not exist.")
        return
    document_chunks = load_documents_from_folder(document_folder, sentence_transformer_model)
    if not document_chunks:
        print("No valid PDF documents found in sample_data folder.")
        return
    for document_name, chunks in document_chunks:
        chroma_collection = add_document_to_collection(chroma_collection, chunks, document_name)

    # Summarize collection
    print(summarize_collection(chroma_collection))

    # User interaction loop
    print("\n" + "="*60)
    print("RAG SYSTEM WITH CITATION SUPPORT")
    print("="*60)
    print("Instructions:")
    print("- Ask any question about your documents")
    print("- After each answer, you'll see the source documents")
    print("- Press 'o' to open any PDF source, or Enter to continue")
    print("- Type 'bye' to exit")
    print("="*60)

    while True:
        question = input("\nPlease enter your question, or type 'bye' to exit: ")
        if question.lower() == "bye":
            print("Goodbye!")
            break
        generateAnswer(RAG_LLM, chroma_collection, question, document_folder, only_response=False)

if __name__ == "__main__":
    main()