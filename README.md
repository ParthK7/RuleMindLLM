# RuleMindLLM

**RuleMindLLM** is a local language model built using **Ollama** and **ChromaDB** that answers rule-based questions about popular games like **Uno**, **Monopoly**, and **Poker**.

It uses **Retrieval-Augmented Generation (RAG)** to provide accurate, grounded responses by leveraging vector similarity search on embedded chunks of official rulebooks.

---

## 🧠 Tech Stack

### 🔹 Ollama (using the **Gemma** model)
- Runs the local LLM efficiently on your machine  
- Handles natural language understanding and response generation  
- Accepts external context passed from ChromaDB to ground responses in documents  

### 🔹 ChromaDB
- Converts game rule PDFs into vector embeddings  
- Stores embeddings for fast similarity search  
- Retrieves the most relevant rule chunks based on user queries  
- Supports RAG: provides context to the LLM at inference time  

---

## 📚 Example Use Case

**Ask:**  
"Can you stack a Draw Two in Uno?"

**→ RuleMindLLM** will return an answer directly from the Uno rulebook PDF, explaining whether stacking is officially allowed.

---

## 🚀 How to Run (WIP)

1. Install [Ollama](https://ollama.com) and run `ollama run gemma`  
2. Clone this repo  
3. Load game rule PDFs → embed → store in ChromaDB  
4. Ask questions via CLI or UI (under development)

---

## 💡 Motivation

LLMs are great at language, but they often hallucinate. **RuleMindLLM** anchors LLMs in **real documents**, making them practical assistants for rule-based domains like games.
