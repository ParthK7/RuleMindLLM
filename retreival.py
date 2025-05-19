from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_chroma import Chroma
from get_embedding_function import get_embedding_function
import argparse
import os
import shutil

CHROMA_PATH = "chroma"

def main():
    # Check if the database should be cleared (using the --reset flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()

    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store.
    documents = load_documents()
    print(f"📄 Loaded {len(documents)} PDF documents")

    chunks = split_documents(documents)
    print(f"🔪 Split into {len(chunks)} chunks")

    add_to_chroma(chunks)

def load_documents():
    loader = PyPDFDirectoryLoader("data")
    return loader.load()

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

embedding_function = get_embedding_function()

def add_to_chroma(chunks: list[Document]):
    db = Chroma(
        collection_name="rag-chroma",
        embedding_function=embedding_function,
        persist_directory=CHROMA_PATH
    )

    chunks_with_ids = calculate_chunk_ids(chunks)

    # Fetch existing IDs
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"📦 Existing documents in DB: {len(existing_ids)}")

    # Filter new chunks
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"🆕 Adding {len(new_chunks)} new documents to DB")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        print("✅ Chroma DB updated and persisted.")
    else:
        print("✅ No new documents to add")

    # Final check: total chunks
    total = len(db.get()["ids"])
    print(f"📊 Total chunks now in DB: {total}")

def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        chunk.metadata["id"] = chunk_id

    return chunks

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

if __name__ == "__main__":
    main()
