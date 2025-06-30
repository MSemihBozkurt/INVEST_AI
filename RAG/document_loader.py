import os
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter

def convert_PDF_Text(pdf_path):
    """Extract text from a PDF file with better text cleaning."""
    try:
        reader = PdfReader(pdf_path)
        pdf_texts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                # Clean up the text - remove excessive whitespace and fix spacing
                cleaned_text = ' '.join(text.split())
                pdf_texts.append(cleaned_text)
        print(f"Document: {pdf_path}, pages processed: {len(pdf_texts)}")
        return pdf_texts
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return []

def convert_Page_ChunkinChar(pdf_texts, chunk_size=2000, chunk_overlap=200):
    """Split text into character-based chunks with better overlap."""
    character_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    character_split_texts = character_splitter.split_text('\n\n'.join(pdf_texts))
    print(f"Total number of chunks (split by max char = {chunk_size}): {len(character_split_texts)}")
    return character_split_texts

def convert_Chunk_Token(text_chunksinChar, sentence_transformer_model, chunk_overlap=20, tokens_per_chunk=120):
    """Split character-based chunks into token-based chunks with better settings."""
    token_splitter = SentenceTransformersTokenTextSplitter(
        chunk_overlap=chunk_overlap,
        model_name=sentence_transformer_model,
        tokens_per_chunk=tokens_per_chunk
    )
    text_chunksinTokens = []
    for text in text_chunksinChar:
        text_chunksinTokens += token_splitter.split_text(text)
    print(f"Total number of chunks (split by {tokens_per_chunk} tokens per chunk): {len(text_chunksinTokens)}")
    return text_chunksinTokens

def load_documents_from_folder(folder_path, sentence_transformer_model):
    """Load and process all PDFs from the specified folder."""
    text_chunks = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.pdf'):
            pdf_path = os.path.join(folder_path, file_name)
            pdf_texts = convert_PDF_Text(pdf_path)
            if pdf_texts:
                text_chunksinChar = convert_Page_ChunkinChar(pdf_texts)
                text_chunksinTokens = convert_Chunk_Token(text_chunksinChar, sentence_transformer_model)
                text_chunks.extend([(file_name, text_chunksinTokens)])
    return text_chunks