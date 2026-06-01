import time

from ollama import chat

start = time.time()

response = chat(
    model="gemma3:1b",
    messages=[{"role": "user", "content": "Halo"}],
)

print("Generation:", time.time() - start)
