import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split

class CombinedQADataset(Dataset):
    def __init__(self, data, pad_token_id):
        self.data = data
        self.pad_token_id = pad_token_id
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_ids = torch.tensor(self.data.iloc[idx]['input_ids'], dtype=torch.long)
        target_ids = torch.tensor(self.data.iloc[idx]['target_ids'], dtype=torch.long)
        attention_mask = (input_ids != self.pad_token_id).long()
        return {
            'input_ids': input_ids,
            'labels': target_ids,
            'attention_mask': attention_mask
        }

def train_combined_model(data_file, model_save_path, epochs=20, batch_size=4, lr=3e-5):
    data = pd.read_pickle(data_file)
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = tokenizer.pad_token_id
    
    train_data, val_data = train_test_split(data, test_size=0.1)
    
    train_dataset = CombinedQADataset(train_data, pad_token_id)
    val_dataset = CombinedQADataset(val_data, pad_token_id)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    optimizer = AdamW(model.parameters(), lr=lr)
    
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    model.train()
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        train_loss = 0
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
                loss = outputs.loss
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {train_loss}, Validation Loss: {val_loss}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(model_save_path)
            tokenizer.save_pretrained(model_save_path)
            print(f"Model saved at epoch {epoch+1}")

if __name__ == "__main__":
    train_combined_model("combined_preprocessed_data.pkl", "my_combined_gpt2_model")