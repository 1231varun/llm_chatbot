from fastapi import FastAPI
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from fastapi.middleware.cors import CORSMiddleware
import torch
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your trained model and tokenizer
model_path = "llm/my_combined_gpt2_model"
if not os.path.exists(model_path):
    raise ValueError(f"Model path '{model_path}' does not exist")

model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

class Prompt(BaseModel):
    prompt: str

@app.post("/chat")
async def chat(prompt: Prompt):
    inputs = tokenizer.encode(prompt.prompt + tokenizer.eos_token, return_tensors="pt")
    attention_mask = (inputs != tokenizer.pad_token_id).long()
    
    outputs = model.generate(
        inputs, 
        attention_mask=attention_mask,
        max_length=50,
        pad_token_id=tokenizer.pad_token_id,
        top_p=0.9,  # Nucleus sampling
        top_k=0,  # Disable top-k sampling
        temperature=0.7,  # Add temperature to control randomness
        repetition_penalty=1.2,  # Penalty to avoid repetition
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
