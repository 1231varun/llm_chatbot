import React, { useState } from "react";
import axios from "axios";

function App() {
  const [prompt, setPrompt] = useState("");
  const [response, setResponse] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const result = await axios.post("http://localhost:8000/chat", { prompt });
      setResponse(result.data.response);
    } catch (error) {
      console.error("Error fetching response", error);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>LLM Chatbot</h1>
        <form onSubmit={handleSubmit}>
          <input
            type="text"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Enter your prompt"
          />
          <button type="submit">Submit</button>
        </form>
        <div>
          <p>{response}</p>
        </div>
      </header>
    </div>
  );
}

export default App;