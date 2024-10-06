import os
import csv
import fitz
import sys
from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizerFast
import tensorflow as tf
from utils import split_texts_into_blocks, process_windows, add_custom_tokens

def extract_features(pdf_path, test_csv='test.csv'):
    print("Starting extract_features")
    try:
        doc = fitz.open(pdf_path)
        texts = []
        labels = []
        with open(test_csv, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            label_list = [row[0] for row in reader]
        label_index = 0
        blocks_all = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            blocks = page.get_text("blocks")
            for block in blocks:
                if len(block) < 6:
                    print(f"Skipping incomplete block on page {page_num}")
                    continue
                x0, y0, x1, y1, text, block_id = block[:6]
                if text.strip():
                    blocks_all.append(block)
                    if label_index < len(label_list):
                        try:
                            label = int(label_list[label_index])
                        except ValueError:
                            print(f"Invalid label at index {label_index}: {label_list[label_index]}")
                            label = 0
                        labels.append(label)
                        texts.append(text.replace('\n', ' '))
                        label_index += 1
        print(f"extract_features completed with {len(texts)} texts")
        return texts, labels, blocks_all
    except Exception as e:
        print(f"Error in extract_features: {e}")
        raise

def prepare_inputs(texts, tokenizer):
    print("Starting prepare_inputs")
    try:
        processed_input_ids = []
        for i, text in enumerate(texts):
            blocks, _ = split_texts_into_blocks([text], None, tokenizer, max_tokens=60)
            input_ids, _ = process_windows(blocks, None, tokenizer)
            for idx, ids in enumerate(input_ids):
                length = len(ids)
                print(f"Text {i}, Window {idx}: Length {length}")
                if length > 512:
                    print(f"Warning: Text {i}, Window {idx} exceeds 512 tokens with length {length}")
                    print(f"Problematic input_ids: {ids}")
                    decoded_text = tokenizer.decode(ids, skip_special_tokens=True)
                    print(f"Decoded Text: {decoded_text}\n")
                processed_input_ids.append(ids)
        max_length = 512
        input_ids_padded = tf.keras.preprocessing.sequence.pad_sequences(
            processed_input_ids, maxlen=max_length, dtype='int32',
            padding='post', truncating='post', value=tokenizer.pad_token_id
        )
        attention_mask = (input_ids_padded != tokenizer.pad_token_id).astype(int)
        print(f"prepare_inputs completed with padded input_ids shape: {input_ids_padded.shape}")
        for idx, seq in enumerate(input_ids_padded):
            seq_length = sum(seq != tokenizer.pad_token_id)
            if seq_length > 512:
                print(f"Error: Sequence {idx} has length {seq_length} which exceeds 512 tokens.")
                decoded_seq = tokenizer.decode(seq, skip_special_tokens=True)
                print(f"Decoded Sequence {idx}: {decoded_seq}\n")
        return input_ids_padded, attention_mask, processed_input_ids
    except Exception as e:
        print(f"Error in prepare_inputs: {e}")
        raise

def predict_with_debug(pdf_path, output_file, model_dir='distilbert-classifier'):
    print("Starting predict_with_debug")
    try:
        texts, _, blocks_all = extract_features(pdf_path, test_csv='test.csv')
        tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
        print("Tokenizer loaded")
        
        model = TFDistilBertForSequenceClassification.from_pretrained(model_dir)
        print("Model loaded")
        
        input_ids_padded, attention_mask, processed_input_ids = prepare_inputs(texts, tokenizer)
        
        lengths = [sum(ids != tokenizer.pad_token_id) for ids in input_ids_padded]
        max_length = max(lengths) if lengths else 0
        print(f"Maximum input_ids length: {max_length}")
        if max_length > 512:
            print("Error: Some input_ids exceed the maximum sequence length of 512.")
        
        print("Starting batch predictions")
        batch_size = 16
        num_batches = (len(input_ids_padded) + batch_size - 1) // batch_size
        predictions = []
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = start + batch_size
            batch_input_ids = input_ids_padded[start:end]
            batch_attention_mask = attention_mask[start:end]
            print(f"Processing batch {batch_idx + 1}/{num_batches}")
            try:
                batch_logits = model(batch_input_ids, attention_mask=batch_attention_mask).logits
                batch_predictions = tf.argmax(batch_logits, axis=1).numpy()
                predictions.extend(batch_predictions)
                print(f"Batch {batch_idx + 1} predictions: {batch_predictions}")
            except Exception as batch_e:
                print(f"Error in batch {batch_idx + 1}: {batch_e}")
                for i in range(len(batch_input_ids)):
                    seq_length = sum(batch_input_ids[i] != tokenizer.pad_token_id)
                    if seq_length > 512:
                        print(f"  Sequence {start + i} in batch {batch_idx + 1} exceeds 512 tokens with length {seq_length}")
                        decoded_text = tokenizer.decode(batch_input_ids[i], skip_special_tokens=True)
                        print(f"  Decoded Text: {decoded_text}\n")
        
        predicted_labels = predictions
        print(f"Predicted labels: {predicted_labels}")
        
        label_map = {
            0: 'Header',
            1: 'Body',
            2: 'Footer',
            3: 'Quote'
        }
        print("Mapping labels to types")
        
        with open(output_file, "a", encoding='utf-8') as file:
            for block, label in zip(blocks_all, predicted_labels):
                if len(block) < 6:
                    print("Skipping incomplete block during writing output")
                    continue
                x0, y0, x1, y1, text, block_id = block[:6]
                if not text.strip():
                    continue
                block_type = label_map.get(label, 'Unknown')
                block_text = text.replace('\n', ' ').strip()
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
        print("Prediction results written to output file")
    except Exception as e:
        print(f"Error in predict_with_debug: {e}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_infer.py <mode>")
        print("Modes: train, infer")
        sys.exit(1)
    
    mode = sys.argv[1]
    pdf_path = 'input.pdf'
    test_csv = 'test.csv'
    output_dir = 'distilbert-classifier'
    output_file = 'output.txt'
    
    if mode == 'train':
        print("Train mode not implemented in debug_infer.py")
    elif mode == 'infer':
        predict_with_debug(pdf_path, output_file, model_dir=output_dir)
    else:
        print("Invalid mode. Use 'train' or 'infer'.")

if __name__ == "__main__":
    main()

