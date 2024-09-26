import math
from transformers import DistilBertTokenizerFast

START_CENTRAL = '[START]'
END_CENTRAL = '[END]'
CONTEXT_SPLIT = '[CONTEXT]'
DUMMY_TEXT = '[DUMMY]'
DUMMY_LABEL = 0

def split_texts_into_blocks(texts, labels, tokenizer, max_tokens=60):
    blocks = []
    block_labels = [] if labels is not None else None
    for i, text in enumerate(texts):
        tokens = tokenizer.encode(text, add_special_tokens=False)
        total_tokens = len(tokens)
        if total_tokens == 0:
            continue
        num_blocks = math.ceil(total_tokens / max_tokens)
        for j in range(num_blocks):
            start = j * max_tokens
            end = min(start + max_tokens, total_tokens)
            block_tokens = tokens[start:end]
            blocks.append(block_tokens)
            if block_labels is not None:
                block_labels.append(labels[i])
    return blocks, block_labels

def process_windows(blocks, block_labels, tokenizer):
    processed_input_ids = []
    processed_labels = [] if block_labels is not None else None
    total_blocks = len(blocks)
    special_token_ids = tokenizer.convert_tokens_to_ids([
        START_CENTRAL, END_CENTRAL, CONTEXT_SPLIT, DUMMY_TEXT
    ])
    START_CENTRAL_ID, END_CENTRAL_ID, CONTEXT_SPLIT_ID, DUMMY_TEXT_ID = special_token_ids
    for idx in range(total_blocks):
        central_block = [START_CENTRAL_ID] + blocks[idx] + [END_CENTRAL_ID]
        context_before = blocks[max(idx - 3, 0):idx]
        while len(context_before) < 3:
            context_before.insert(0, [DUMMY_TEXT_ID])
        context_before_flat = []
        for block in context_before:
            context_before_flat.extend(block + [CONTEXT_SPLIT_ID])
        if context_before_flat:
            context_before_flat = context_before_flat[:-1]
        context_after = blocks[idx + 1:idx + 4]
        while len(context_after) < 2:
            context_after.append([DUMMY_TEXT_ID])
        context_after_flat = []
        for block in context_after:
            context_after_flat.extend(block + [CONTEXT_SPLIT_ID])
        if context_after_flat:
            context_after_flat = context_after_flat[:-1]
        window_tokens = context_before_flat + central_block + context_after_flat
        #print("[START]", tokenizer.decode(window_tokens),"[END]\n")
        processed_input_ids.append(window_tokens)
        if processed_labels is not None:
            processed_labels.append(block_labels[idx])
    return processed_input_ids, processed_labels

def add_custom_tokens(tokenizer):
    special_tokens_list = [START_CENTRAL, END_CENTRAL, CONTEXT_SPLIT, DUMMY_TEXT]
    tokenizer.add_tokens(special_tokens_list)
    return tokenizer
