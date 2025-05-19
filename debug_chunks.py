# debug_chunks.py
from langchain_community.vectorstores import Chroma
from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

embedding_function = get_embedding_function()
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

# Load all documents
collection = db.get()
print(f"Total chunks in DB: {len(collection['documents'])}")

# Print chunks that mention 'jail'
for doc, id_ in zip(collection["documents"], collection["ids"]):
    if "jail" in doc.lower():
        print(f"\n🧩 Chunk ID: {id_}")
        print(doc)
