import os
import csv
import fitz

def extract_features(pdf_path, test_csv='test.csv', output_txt='text_labels.txt'):
    if os.path.exists(output_txt):
        print(f"Loading text and labels from {output_txt}")
        texts, labels = [], []
        with open(output_txt, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|||')
                if len(parts) != 2:
                    print(f"Malformed line: {line.strip()}")
                    texts.append("Malformed line")
                    labels.append(int(4))
                text, label = parts
                texts.append(text)
                labels.append(int(label))
        return texts, labels

    print("Extracting text blocks from PDF and reading labels from CSV...")
    doc = fitz.open(pdf_path)
    texts = []
    labels = []

    with open(test_csv, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        label_list = [row[0] for row in reader]

    label_index = 0
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        blocks = page.get_text("blocks")
        for block in blocks:
            x0, y0, x1, y1, text, block_id = block[:6]
            if text.strip():
                if label_index >= len(label_list):
                    raise ValueError("Not enough labels in test.csv for the extracted text blocks.")
                label = int(label_list[label_index])
                sanitized_text = text.replace('\n', ' ').replace('|||', ' ')
                texts.append(sanitized_text)
                labels.append(label)
                label_index += 1

    if label_index != len(label_list):
        raise ValueError("Number of labels in test.csv does not match the number of text blocks extracted.")

    with open(output_txt, 'w', encoding='utf-8') as f:
        for text, label in zip(texts, labels):
            f.write(f"{text}|||{label}\n")

    return texts, labels

if __name__ == "__main__":
    extract_features("input.pdf")

