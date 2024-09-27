import math
import csv
import fitz
from transformers import DistilBertTokenizerFast

START_CENTRAL = '[START]'
END_CENTRAL = '[END]'
CONTEXT_SPLIT = '[CONTEXT]'
DUMMY_TEXT = '[DUMMY]'
DUMMY_LABEL = 0

def extract_features(pdf_path, test_csv='test.csv'):
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
            x0, y0, x1, y1, text, block_id = block[:6]
            if text.strip():
                blocks_all.append(block)
                if label_index < len(label_list):
                    label = int(label_list[label_index])
                    labels.append(label)
                    texts.append(text.replace('\n', ' '))
                    label_index += 1
    return texts, labels, blocks_all

def split_texts_into_blocks(texts, tokenizer):
    return [block_tokens for block_tokens, _ in _split_into_blocks(texts, tokenizer)]

def split_labels_into_blocks(labels, texts, tokenizer):
    blocks_info = _split_into_blocks(texts, tokenizer)
    block_labels = []
    for i, text in enumerate(texts):
        tokens = tokenizer.encode(text, add_special_tokens=False)
        num_blocks = math.ceil(len(tokens) / 60)
        block_labels.extend([labels[i]] * num_blocks)
    return block_labels

def _split_into_blocks(texts, tokenizer, max_tokens=60):
    blocks_info = []
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        total_tokens = len(tokens)
        if total_tokens == 0:
            continue
        num_blocks = math.ceil(total_tokens / max_tokens)
        for j in range(num_blocks):
            start = j * max_tokens
            end = min(start + max_tokens, total_tokens)
            block_tokens = tokens[start:end]
            blocks_info.append((block_tokens, num_blocks))
    return blocks_info

def process_windows(blocks, block_labels, tokenizer):
    context_size = 1  
    processed_input_ids = []
    processed_labels = [] if block_labels is not None else None
    total_blocks = len(blocks)
    special_token_ids = tokenizer.convert_tokens_to_ids([START_CENTRAL, END_CENTRAL, CONTEXT_SPLIT, DUMMY_TEXT])
    START_CENTRAL_ID, END_CENTRAL_ID, CONTEXT_SPLIT_ID, DUMMY_TEXT_ID = special_token_ids
    for idx in range(total_blocks):
        central_block = [START_CENTRAL_ID] + blocks[idx] + [END_CENTRAL_ID]
        context_before = blocks[max(idx - context_size, 0):idx]
        while len(context_before) < context_size:
            context_before.insert(0, [DUMMY_TEXT_ID])
        context_before_flat = []
        for block in context_before:
            context_before_flat.extend(block + [CONTEXT_SPLIT_ID])
        if context_before_flat:
            context_before_flat = context_before_flat[:-1]  
        context_after = blocks[idx + 1:idx + 1 + context_size]
        while len(context_after) < context_size:
            context_after.append([DUMMY_TEXT_ID])
        context_after_flat = []
        for block in context_after:
            context_after_flat.extend(block + [CONTEXT_SPLIT_ID])
        if context_after_flat:
            context_after_flat = context_after_flat[:-1]  
        window_tokens = context_before_flat + central_block + context_after_flat
        processed_input_ids.append(window_tokens)
        if processed_labels is not None:
            processed_labels.append(block_labels[idx])
    return processed_input_ids, processed_labels

def add_custom_tokens(tokenizer):
    special_tokens_list = [START_CENTRAL, END_CENTRAL, CONTEXT_SPLIT, DUMMY_TEXT]
    tokenizer.add_tokens(special_tokens_list)
    return tokenizer

