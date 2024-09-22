# use_model.py
import os
import fitz  # PyMuPDF
import tensorflow as tf
from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizerFast

def drop_to_file(block_text, block_type):
    with open("output.txt", "a", encoding='utf-8') as file:
        if block_type == 'Header':
            file.write(f"<h1>{block_text}</h1>\n\n")
        elif block_type == 'Body':
            file.write(f"<body>{block_text}</body>\n\n")
        elif block_type == 'Footer':
            file.write(f"<footer>{block_text}</footer>\n\n")
        elif block_type == 'Quote':
            file.write(f"<blockquote>{block_text}</blockquote>\n\n")
        else:
            file.write(f"{block_text}\n\n")

def classify_text(model, tokenizer, text):
    encodings = tokenizer(text, return_tensors='tf', truncation=True, padding='max_length', max_length=512)
    logits = model(encodings['input_ids'], attention_mask=encodings['attention_mask']).logits
    prediction = tf.argmax(logits, axis=1).numpy()[0]
    certainty = tf.nn.softmax(logits, axis=1)[0][prediction].numpy()
    block_types = ['Header', 'Body', 'Footer', 'Quote']
    predicted_block_type = block_types[prediction]
    return predicted_block_type, certainty

def main():
    pdf_path = "input.pdf"
    model_path = 'distilbert-classifier'

    if not os.path.exists(model_path):
        print(f"Model path '{model_path}' does not exist. Please train the model first.")
        return

    print("Loading tokenizer and model...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    model = TFDistilBertForSequenceClassification.from_pretrained(model_path)

    open("output.txt", "w").close()

    print("Opening PDF and classifying text blocks...")
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        blocks = page.get_text("blocks")
        for block in blocks:
            x0, y0, x1, y1, text, block_id = block[:6]
            if text.strip():
                block_type, certainty = classify_text(model, tokenizer, text)
                drop_to_file(text, block_type)
                print(f"Page {page_num + 1}: Classified block as {block_type} with certainty {certainty:.2f}")

    print("Classification complete. Output saved to 'output.txt'.")

if __name__ == "__main__":
    main()

