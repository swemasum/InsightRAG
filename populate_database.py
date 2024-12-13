import argparse
import os
import shutil
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain.vectorstores.chroma import Chroma

# Define paths for data and database storage
CHROMA_DATABASE_DIRECTORY = "chroma"
PDF_DATA_DIRECTORY = "data"

def main():
    """
    Main function to manage the document processing pipeline.
    """
    # Parse command-line arguments to check if the database should be reset
    parser = argparse.ArgumentParser(description="Process PDF documents into embeddings.")
    parser.add_argument("--reset", action="store_true", help="Reset the database before processing.")
    args = parser.parse_args()

    if args.reset:
        print("✨ Clearing Database")
        reset_chroma_database()

    # Load, split, and store documents in the database
    pdf_documents = load_pdf_documents()
    document_chunks = split_documents_into_chunks(pdf_documents)
    store_chunks_in_chroma(document_chunks)

def load_pdf_documents():
    """
    Load PDF documents from the specified data directory.

    Returns:
        List of loaded documents.
    """
    pdf_loader = PyPDFDirectoryLoader(PDF_DATA_DIRECTORY)
    return pdf_loader.load()

def split_documents_into_chunks(documents: list[Document]):
    """
    Split loaded documents into manageable text chunks.

    Args:
        documents: List of documents to split.

    Returns:
        List of document chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,           # Maximum size of each chunk
        chunk_overlap=80,         # Overlap between chunks to maintain context
        length_function=len,      # Function to measure text length
        is_separator_regex=False  # Indicates if separators are regex patterns
    )
    return text_splitter.split_documents(documents)

def store_chunks_in_chroma(chunks: list[Document]):
    """
    Store document chunks into the Chroma database, avoiding duplicates.

    Args:
        chunks: List of document chunks to store.
    """
    # Initialize Chroma database
    chroma_db = Chroma(
        persist_directory=CHROMA_DATABASE_DIRECTORY, 
        embedding_function=get_embedding_function()
    )

    # Assign unique IDs to each chunk
    chunks_with_ids = assign_unique_ids_to_chunks(chunks)

    # Retrieve existing document IDs from the database
    existing_documents = chroma_db.get(include=[])
    existing_ids = set(existing_documents["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Identify and store new chunks
    new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]

    if new_chunks:
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        chroma_db.add_documents(new_chunks, ids=new_chunk_ids)
        chroma_db.persist()
    else:
        print("✅ No new documents to add")

def assign_unique_ids_to_chunks(chunks):
    """
    Assign unique IDs to each document chunk based on source, page, and chunk index.

    Args:
        chunks: List of document chunks to assign IDs.

    Returns:
        List of chunks with assigned IDs.
    """
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # Increment index if on the same page, otherwise reset
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Assign a unique ID and update metadata
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        chunk.metadata["id"] = chunk_id
        last_page_id = current_page_id

    return chunks

def reset_chroma_database():
    """
    Clear the existing Chroma database by deleting its directory.
    """
    if os.path.exists(CHROMA_DATABASE_DIRECTORY):
        shutil.rmtree(CHROMA_DATABASE_DIRECTORY)

if __name__ == "__main__":
    main()
