import os
import time

import chromadb
from dotenv import load_dotenv

# from ollama import chat
from openai import OpenAI
from sentence_transformers import SentenceTransformer

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

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
        n_results=5,
        include=["documents", "metadatas", "distances"],
    )

    print("Retrieval:", time.time() - start)
    print(results["distances"][0])

    best_distance = results["distances"][0][0]
    print("Best Distance:", best_distance)
    if best_distance > 0.35:
        print("Informasi tidak ditemukan dalam dokumen.")
        continue

    contexts = []

    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        contexts.append(
            f"""
    Source: {meta["source"]}
    Chunk ID: {meta["chunk_id"]}

    {doc}
    """
        )

    context = "\n\n".join(contexts)

    prompt = f"""
Anda adalah asisten akademik ITB.

Gunakan hanya informasi dari context.

Sebutkan pasal jika tersedia.

Jika jawaban tidak ditemukan dalam context,
katakan bahwa informasi tidak tersedia.

Context:
{context}

Pertanyaan:
{question}

Jawaban:
"""

    start = time.time()

    response = client.chat.completions.create(
        model="google/gemini-2.5-flash",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    answer = response.choices[0].message.content
    print("Generation:", time.time() - start)

    #    print()
    print(answer)
#    print(response["message"]["content"])
