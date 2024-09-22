# train_model.py
import numpy as np
from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizerFast
from sklearn.model_selection import train_test_split
import tensorflow as tf

def load_data(input_txt='text_labels.txt'):
    texts, labels = [], []
    with open(input_txt, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|||')
            if len(parts) != 2:
                print(f"Skipping malformed line: {line.strip()}")
                continue
            text, label = parts
            texts.append(text)
            labels.append(int(label))
    return texts, labels

def main():
    texts, labels = load_data()

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    # After tokenizing and converting to NumPy arrays
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=512, return_tensors='tf')

    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']
    labels = tf.convert_to_tensor(labels)

    # Create a TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((
        {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        },
        labels
    ))

    # Shuffle and split the dataset
    dataset = dataset.shuffle(buffer_size=10000, seed=42)
    train_size = int(0.9 * len(texts))
    train_dataset = dataset.take(train_size).batch(8)
    val_dataset = dataset.skip(train_size).batch(8)

    # Compile the model
    model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=4)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    # Fit the model using the TensorFlow datasets
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=3
    )

    # Save the trained model and tokenizer
    model.save_pretrained('distilbert-classifier')
    tokenizer.save_pretrained('distilbert-classifier')
    print("Model trained and saved to 'distilbert-classifier'.")

if __name__ == "__main__":
    main()

