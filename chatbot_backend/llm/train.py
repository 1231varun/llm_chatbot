import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset
from tqdm import tqdm

class CombinedQADataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        input_text = example["input"]
        target_text = example["output"]

        # Tokenize and truncate/pad to max_length
        encoded = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"]

        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0),
        }

def get_device():
    """Determine the best device for training."""
    if torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon GPU)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    else:
        print("Using CPU")
        return torch.device("cpu")

def train_combined_model(dataset_name, model_save_path, max_length=512, epochs=5, batch_size=4, lr=1e-5, accumulation_steps=4):
    # Load dataset
    dataset = load_dataset(dataset_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")

    # Add special tokens
    tokenizer.add_special_tokens({
        "additional_special_tokens": ["<USER>", "<BOT>"],
        "pad_token": "[PAD]"
    })

    # Prepare dataset
    def preprocess(example):
        dialog = example["dialog"]
        input_text = "<USER> " + " <BOT> ".join(dialog[:-1])  # Combine dialog turns
        target_text = "<BOT> " + dialog[-1]  # Bot's response
        return {
            "input": input_text,
            "output": target_text,
        }

    processed_dataset = dataset.map(preprocess, remove_columns=dataset["train"].column_names)
    train_data = processed_dataset["train"]
    val_data = processed_dataset["validation"] if "validation" in processed_dataset else None

    train_dataset = CombinedQADataset(train_data, tokenizer, max_length=max_length)
    val_dataset = CombinedQADataset(val_data, tokenizer, max_length=max_length) if val_data else None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True) if val_dataset else None

    # Load model
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    model.resize_token_embeddings(len(tokenizer))
    device = get_device()
    model.to(device)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs // accumulation_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Mixed precision setup
    scaler = GradScaler(enabled=torch.cuda.is_available())

    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        for step, batch in enumerate(progress_bar):
            optimizer.zero_grad()
            inputs = {key: value.to(device) for key, value in batch.items()}

            # Mixed precision training
            with autocast(enabled=torch.cuda.is_available()):
                outputs = model(**inputs)
                loss = outputs.loss / accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            train_loss += loss.item() * accumulation_steps
            progress_bar.set_postfix({"Batch Loss": loss.item() * accumulation_steps})

        train_loss /= len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {train_loss}")

        # Validation
        if val_loader:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    inputs = {key: value.to(device) for key, value in batch.items()}
                    with autocast(enabled=torch.cuda.is_available()):
                        outputs = model(**inputs)
                        val_loss += outputs.loss.item()

            val_loss /= len(val_loader)
            print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model.save_pretrained(model_save_path)
                tokenizer.save_pretrained(model_save_path)
                print(f"Model saved at epoch {epoch + 1}")

    print("Training completed!")

if __name__ == "__main__":
    train_combined_model("daily_dialog", "my_combined_gpt2_model", max_length=512)
