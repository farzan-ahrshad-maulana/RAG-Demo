import fitz
import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt")
nltk.download("punkt_tab")


def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


text = extract_text("data/Buku_Peraturan_Akademik_2024_PR_25A.pdf")

sentences = sent_tokenize(text)
print("Jumlah kalimat:", len(sentences))
print()
print(sentences[0])

print()
print(sentences[1])

print()
print(sentences[2])
print()
print(sentences[0])
