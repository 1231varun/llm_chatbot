# Chatbot LLM Project

This project implements a chatbot using a GPT-2 language model. The chatbot is trained on a combined dataset of question-answer pairs and dialogue exchanges, enabling it to handle various conversational contexts. The backend is built using FastAPI, and the frontend is a simple ReactJS application.

As of now this is day 1 effort that's committed as initial commit. Its still not able to generate correct responses and has still a long way to go.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training the Model](#training-the-model)
- [Running the Backend](#running-the-backend)
- [Running the Frontend](#running-the-frontend)
- [Testing the Chatbot](#testing-the-chatbot)
- [Acknowledgments](#acknowledgments)

## Requirements

- Python 3.8 or higher
- Node.js and npm
- pip (Python package installer)

## Installation

1. **Clone the repository:**

    ```sh
    git clone https://github.com/yourusername/chatbot-llm.git
    cd chatbot-llm
    ```

2. **Set up the Python environment:**

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3. **Set up the ReactJS frontend:**

    ```sh
    cd llm_frontend
    npm install
    cd ..
    ```

## Data Preparation

1. **Create your datasets:**

    - `qa_dataset.json` (example):
    
    ```json
    [
        {"question": "What is AI?", "answer": "AI stands for Artificial Intelligence."},
        {"question": "Who wrote '1984'?", "answer": "George Orwell wrote '1984'."}
    ]
    ```

    - `dialogue_dataset.json` (example):
    
    ```json
    [
        {"context": "Hello! How can I help you today?", "response": "Hi! I'm looking for information on your services."},
        {"context": "Sure, what do you need help with?", "response": "Can you tell me more about your pricing plans?"}
    ]
    ```

2. **Preprocess the datasets:**

    ```sh
    python preprocess.py
    ```

## Training the Model

Train the GPT-2 model on the combined dataset:

```sh
python train.py
```

## Running the Backend

1. Ensure the virtual environment is activated:

```sh
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

2. Start the FastAPI server:
```sh
uvicorn main:app --reload
```

## Running the Frontend

1. Navigate to the frontend directory:
```sh
cd llm_frontend
```

2. Start the React application:
```sh
npm start
```

## Testing the Chatbot

1. Open your browser and navigate to http://localhost:3000.
2. Enter a prompt in the input field and click "Submit".
3. The chatbot should respond based on the trained model.
