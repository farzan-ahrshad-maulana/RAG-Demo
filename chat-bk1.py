import time

import chromadb
from ollama import chat
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("intfloat/multilingual-e5-small")

chroma_client = chromadb.PersistentClient(path="./chroma.db")

collection = chroma_client.get_collection("itb_regulations")


def get_embedding(text):
    return embedding_model.encode(text).tolist()


while True:
    question = input("\nPertanyaan (ketik exit untuk keluar): ")

    if question.lower() == "exit":
        break

    start = time.time()

    question_embedding = get_embedding(question)

    print("Embedding:", time.time() - start)

    start = time.time()

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=3,
    )

    print("Retrieval:", time.time() - start)

    context = "\n\n".join(results["documents"][0])

    prompt = f"""
Jawablah pertanyaan berdasarkan context berikut.

Context:
{context}

Pertanyaan:
{question}
"""

    start = time.time()

    response = chat(
        model="gemma3:1b",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )
    print("Generation:", time.time() - start)

    print()
    print(response["message"]["content"])
