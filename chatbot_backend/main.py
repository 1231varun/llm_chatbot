from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
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
model_path = "llm/my_combined_gpt2_model"  # Adjusted path
if not os.path.exists(model_path):
    raise ValueError(f"Model path '{model_path}' does not exist")

model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

class Prompt(BaseModel):
    prompt: str

@app.post("/chat")
async def chat(prompt: Prompt):
    inputs = tokenizer.encode(f"<USER> {prompt.prompt} <BOT>", return_tensors="pt")
    outputs = model.generate(
        inputs,
        max_length=50,
        pad_token_id=tokenizer.pad_token_id,
        top_p=0.9,
        top_k=50,
        temperature=0.7,
        repetition_penalty=1.5,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
