import csv
import re

def load_test_labels(test_csv='test.csv'):
    labels = []
    with open(test_csv, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row:
                labels.append(int(row[0]))
    return labels

def load_predicted_labels(output_txt='output.txt'):
    block_types = []
    texts = []
    with open(output_txt, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith("<h1>"):
                block_types.append(0)  # Header
                text = re.sub(r'^<h1>(.*?)</h1>\n\n$', r'\1', line.strip())
                texts.append(text)
            elif line.startswith("<body>"):
                block_types.append(1)  # Body
                text = re.sub(r'^<body>(.*?)</body>\n\n$', r'\1', line.strip())
                texts.append(text)
            elif line.startswith("<footer>"):
                block_types.append(2)  # Footer
                text = re.sub(r'^<footer>(.*?)</footer>\n\n$', r'\1', line.strip())
                texts.append(text)
            elif line.startswith("<blockquote>"):
                block_types.append(3)  # Quote
                text = re.sub(r'^<blockquote>(.*?)</blockquote>\n\n$', r'\1', line.strip())
                texts.append(text)
            else:
                continue
    return texts, block_types

def calculate_accuracy(true_labels, predicted_labels):
    if len(true_labels) != len(predicted_labels):
        print(f"Mismatch in number of labels: {len(true_labels)} true vs {len(predicted_labels)} predicted")
        min_len = min(len(true_labels), len(predicted_labels))
        true_labels = true_labels[:min_len]
        predicted_labels = predicted_labels[:min_len]
    correct = sum(t == p for t, p in zip(true_labels, predicted_labels))
    accuracy = (correct / len(true_labels)) * 100
    return accuracy

def main():
    true_labels = load_test_labels()
    texts, predicted_labels = load_predicted_labels()
    accuracy = calculate_accuracy(true_labels, predicted_labels)
    print("\nMisclassified Blocks:")
    block_types = ['Header', 'Body', 'Footer', 'Quote']
    misclassified = False
    for i, (true, pred, text) in enumerate(zip(true_labels, predicted_labels, texts)):
        if true != pred:
            misclassified = True
            print(f"\nBlock {i+1}:")
            print(f"Correct Label: {block_types[true]}")
            print(f"Predicted Label: {block_types[pred]}")
            print(f"Text: {text}")
    print(f"Model Accuracy: {accuracy:.2f}%")
    if not misclassified:
        print("None. All blocks were classified correctly.")

if __name__ == "__main__":
    main()
