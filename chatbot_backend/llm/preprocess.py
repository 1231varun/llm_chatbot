# import json
# import pandas as pd
# from transformers import GPT2Tokenizer

# def preprocess_data(file_path):
#     with open(file_path, 'r') as file:
#         data = json.load(file)
    
#     tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
#     # Set pad token if it does not exist
#     if tokenizer.pad_token_id is None:
#         tokenizer.pad_token = tokenizer.eos_token

#     tokenized_data = []
#     for pair in data:
#         input_text = pair["question"]
#         target_text = pair["answer"]
#         input_ids = tokenizer.encode(input_text + tokenizer.eos_token, add_special_tokens=False)
#         target_ids = tokenizer.encode(target_text + tokenizer.eos_token, add_special_tokens=False)
#         tokenized_data.append((input_ids, target_ids))

#     return tokenized_data

# def pad_sequences(sequences, pad_token_id, max_len=None):
#     if max_len is None:
#         max_len = max(len(seq) for seq in sequences)
#     padded_sequences = []
#     for seq in sequences:
#         if len(seq) < max_len:
#             seq = seq + [pad_token_id] * (max_len - len(seq))
#         else:
#             seq = seq[:max_len]
#         padded_sequences.append(seq)
#     return padded_sequences, max_len

# if __name__ == "__main__":
#     tokenized_data = preprocess_data("dataset.json")
    
#     input_ids = [x[0] for x in tokenized_data]
#     target_ids = [x[1] for x in tokenized_data]
    
#     tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#     pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

#     input_ids, max_len = pad_sequences(input_ids, pad_token_id)
#     target_ids, _ = pad_sequences(target_ids, pad_token_id, max_len)
    
#     df = pd.DataFrame({'input_ids': input_ids, 'target_ids': target_ids})
#     df.to_pickle("preprocessed_data.pkl")

import json
import pandas as pd
from transformers import GPT2Tokenizer

def preprocess_data(file_path, is_qa=True):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenized_data = []
    for pair in data:
        if is_qa:
            input_text = pair["question"]
            target_text = pair["answer"]
        else:
            input_text = pair["context"]
            target_text = pair["response"]
        input_ids = tokenizer.encode(input_text + tokenizer.eos_token, add_special_tokens=False)
        target_ids = tokenizer.encode(target_text + tokenizer.eos_token, add_special_tokens=False)
        tokenized_data.append((input_ids, target_ids))

    return tokenized_data

def pad_sequences(sequences, pad_token_id, max_len=None):
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        if len(seq) < max_len:
            seq = seq + [pad_token_id] * (max_len - len(seq))
        else:
            seq = seq[:max_len]
        padded_sequences.append(seq)
    return padded_sequences, max_len

if __name__ == "__main__":
    qa_data = preprocess_data("qa_dataset.json", is_qa=True)
    dialogue_data = preprocess_data("dialogue_dataset.json", is_qa=False)
    
    combined_data = qa_data + dialogue_data
    
    input_ids = [x[0] for x in combined_data]
    target_ids = [x[1] for x in combined_data]
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    input_ids, max_len = pad_sequences(input_ids, pad_token_id)
    target_ids, _ = pad_sequences(target_ids, pad_token_id, max_len)
    
    df = pd.DataFrame({'input_ids': input_ids, 'target_ids': target_ids})
    df.to_pickle("combined_preprocessed_data.pkl")
