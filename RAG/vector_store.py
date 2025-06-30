import os
import chromadb
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings

def create_chroma_client(collection_name, embedding_function, chromaDB_path):
    """Initialize a persistent ChromaDB client and collection."""
    if not os.path.exists(chromaDB_path):
        os.makedirs(chromaDB_path)
    chroma_client = chromadb.PersistentClient(
        path=chromaDB_path,
        settings=Settings(),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE
    )
    chroma_collection = chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function
    )
    return chroma_client, chroma_collection

def document_exists(chroma_collection, document_name):
    """Check if a document already exists in the collection."""
    if chroma_collection.count() == 0:
        return False
    existing_docs = chroma_collection.get(include=["metadatas"])["metadatas"]
    return any(metadata["document"] == document_name for metadata in existing_docs)

def add_document_to_collection(chroma_collection, text_chunks, document_name, category="Journal Paper"):
    """Add document chunks to the ChromaDB collection with metadata, if not already present."""
    if document_exists(chroma_collection, document_name):
        print(f"Document {document_name} already exists in collection. Skipping insertion.")
        return chroma_collection
    current_count = chroma_collection.count()
    ids = [str(i + current_count) for i in range(len(text_chunks))]
    metadatas = [{"document": document_name, "category": category} for _ in range(len(text_chunks))]
    print(f"Before inserting {document_name}, collection size: {chroma_collection.count()}")
    chroma_collection.add(ids=ids, metadatas=metadatas, documents=text_chunks)
    print(f"After inserting {document_name}, collection size: {chroma_collection.count()}")
    return chroma_collection

def retrieveDocs(chroma_collection, query, n_results=10, return_only_docs=False):
    """Retrieve documents from ChromaDB based on query similarity with metadata for citations."""
    results = chroma_collection.query(
        query_texts=[query],
        include=["documents", "metadatas", "distances"],
        n_results=n_results
    )
    
    if return_only_docs:
        return results['documents'][0] if results and 'documents' in results else []
    
    return results

def search_exact_term(chroma_collection, term):
    """Search for chunks that contain the exact term."""
    all_docs = chroma_collection.get(include=["documents", "metadatas"])
    matching_chunks = []
    
    for i, doc in enumerate(all_docs['documents']):
        if term.lower() in doc.lower():
            matching_chunks.append({
                'document': doc,
                'metadata': all_docs['metadatas'][i]
            })
    
    return matching_chunks

def get_document_sources(results):
    """Extract unique document sources from query results."""
    if not results or 'metadatas' not in results:
        return []
    
    sources = set()
    for metadata_list in results['metadatas']:
        for metadata in metadata_list:
            if 'document' in metadata:
                sources.add(metadata['document'])
    
    return list(sources)

def summarize_collection(chroma_collection):
    """Summarize the collection's contents."""
    summary = []
    summary.append(f"Collection name: {chroma_collection.name}")
    summary.append(f"Number of document chunks in collection: {chroma_collection.count()}")
    distinct_documents = set()
    if chroma_collection.count() > 0:
        try:
            all_metadata = chroma_collection.get(include=["metadatas"])['metadatas']
            for metadata in all_metadata:
                document_name = metadata.get("document", "Unknown")
                distinct_documents.add(document_name)
        except Exception as e:
            print(f"Error getting metadata: {e}")
    
    summary.append("Documents:")
    for document_name in distinct_documents:
        summary.append(document_name)
    return "\n".join(summary)

def clear_collection(chroma_collection):
    """Clear all documents from the collection."""
    if chroma_collection.count() > 0:
        all_ids = chroma_collection.get()["ids"]
        chroma_collection.delete(ids=all_ids)
    print(f"Collection {chroma_collection.name} cleared. Current size: {chroma_collection.count()}")