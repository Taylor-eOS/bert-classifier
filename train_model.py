import os
import sys
import numpy as np
import tensorflow as tf
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
from utils import extract_features, split_texts_into_blocks, split_labels_into_blocks, process_windows, add_custom_tokens

def prepare_dataset(texts, labels, tokenizer):
    processed_input_ids = []
    processed_labels = []
    for i, text in enumerate(texts):
        blocks = split_texts_into_blocks([text], tokenizer)
        block_labels = split_labels_into_blocks([labels[i]], [text], tokenizer)
        input_ids, lbls = process_windows(blocks, block_labels, tokenizer)
        processed_input_ids.extend(input_ids)
        processed_labels.extend(lbls)
    max_length = 512
    input_ids_padded = tf.keras.preprocessing.sequence.pad_sequences(
        processed_input_ids, maxlen=max_length, dtype='int32',
        padding='post', truncating='post', value=tokenizer.pad_token_id
    )
    attention_mask = (input_ids_padded != tokenizer.pad_token_id).astype(int)
    labels_tensor = tf.convert_to_tensor(processed_labels)
    dataset = tf.data.Dataset.from_tensor_slices((
        {
            'input_ids': input_ids_padded,
            'attention_mask': attention_mask
        },
        labels_tensor
    ))
    return dataset

def prepare_inputs(texts, tokenizer):
    try:
        processed_input_ids = []
        for i, text in enumerate(texts):
            blocks = split_texts_into_blocks([text], tokenizer)
            input_ids, _ = process_windows(blocks, None, tokenizer)
            num_sequences = len(input_ids)
            print(f"Text {i}, Number of Windows: {num_sequences}")
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

def train(pdf_path, test_csv, output_dir):
    texts, labels, _ = extract_features(pdf_path, test_csv)
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    tokenizer = add_custom_tokens(tokenizer)
    model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=4)
    model.resize_token_embeddings(len(tokenizer))
    dataset = prepare_dataset(texts, labels, tokenizer)
    dataset = dataset.shuffle(buffer_size=10000, seed=42)
    train_size = int(0.9 * len(texts))
    train_dataset = dataset.take(train_size).batch(8)
    val_dataset = dataset.skip(train_size).batch(8)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=1
    )
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

def predict(pdf_path, output_file, model_dir, batch_size=8):
    texts, _, blocks_all = extract_features(pdf_path, test_csv='test.csv')
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    model = TFDistilBertForSequenceClassification.from_pretrained(model_dir)
    input_ids_padded, attention_mask, blocks = prepare_inputs(texts, tokenizer)
    dataset = tf.data.Dataset.from_tensor_slices((input_ids_padded, attention_mask))
    dataset = dataset.batch(batch_size)
    all_predictions = []
    total_sequences = len(input_ids_padded)
    print(f"Total sequences to process: {total_sequences}")
    total_batches = 0
    for batch_num, (batch_input_ids, batch_attention_mask) in enumerate(dataset):
        print(f"Processing batch {batch_num + 1}/{int(np.ceil(total_sequences / batch_size))}")
        logits = model(batch_input_ids, attention_mask=batch_attention_mask).logits
        batch_pred = tf.argmax(logits, axis=1).numpy()
        print(f"Batch {batch_num + 1} predictions: {batch_pred}")
        all_predictions.extend(batch_pred)
        total_batches += 1
    print(f"All batches processed successfully. Total batches: {total_batches}")
    predicted_labels = np.array(all_predictions)
    label_map = {0: 'Header', 1: 'Body', 2: 'Footer', 3: 'Quote'}
    with open(output_file, "a", encoding='utf-8') as file:
        for block, label in zip(blocks_all, predicted_labels):
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
    print("Prediction process completed")

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'train'
    pdf_path = 'input.pdf'
    test_csv = 'test.csv'
    output_dir = 'distilbert-classifier'
    output_file = 'output.txt'
    if mode == 'train':
        train(pdf_path, test_csv, output_dir)
    elif mode == 'infer':
        predict(pdf_path, output_file, output_dir)
    else:
        print("Invalid mode. Use 'train' or 'infer'.")

if __name__ == "__main__":
    main()

