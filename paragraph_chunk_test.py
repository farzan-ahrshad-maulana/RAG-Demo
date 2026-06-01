import fitz


def extract_text(pdf_path):
    doc = fitz.open(pdf_path)

    text = ""

    for page in doc:
        text += page.get_text()

    return text


text = extract_text("data/Buku_Peraturan_Akademik_2024_PR_25A.pdf")

paragraphs = text.split("\n\n")

paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 50]

print("Jumlah paragraph:", len(paragraphs))

print()
print(paragraphs[0])

print("\n" + "=" * 50 + "\n")

print(paragraphs[1])

print("\n" + "=" * 50 + "\n")

print(paragraphs[2])
