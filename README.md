# 🧠 AI Resume Semantic Search System (Endee-Based Architecture)

---

## 📌 Project Overview

The **AI Resume Semantic Search System** is an AI-powered application that enables recruiters to efficiently identify the most relevant candidates from a large pool of resumes using **semantic search and vector embeddings**.

Unlike traditional keyword-based systems, this solution understands the **context and intent** of both queries and resumes, resulting in more accurate and meaningful candidate matching.

---

## ❗ Problem Statement

Recruiters often spend excessive time manually reviewing resumes. Traditional search systems rely on keyword matching, which leads to:

- Missing qualified candidates due to wording differences  
- Irrelevant or inaccurate results  
- Increased hiring time and inefficiency  

---

## 💡 Solution

This project implements a **vector-based semantic search system** using AI embeddings:

- Converts resumes into high-dimensional vector representations  
- Converts user queries into embeddings  
- Uses similarity search (cosine similarity) to retrieve the most relevant resumes  

---

## 🏗️ System Architecture
User Query
↓
Embedding Model (Sentence Transformers)
↓
Vector Store (Simulated Endee)
↓
Similarity Search (Cosine Similarity)
↓
Top Matching Resumes



---

## ⚙️ Use of Endee

This project is designed to use **Endee** as a high-performance vector database for storing and retrieving embeddings.

⚠️ **Important Note:**

Due to system limitations (Docker/virtualization not supported on the local machine), the project uses a **local in-memory vector store** to simulate Endee functionality.

However:

- The architecture is fully compatible with Endee  
- The vector store module can be easily replaced with Endee APIs  
- The retrieval pipeline remains unchanged  

---

## 🚀 Features

- Upload multiple resumes in PDF format  
- Automatic text extraction from resumes  
- Embedding generation using pre-trained NLP models  
- Semantic search using natural language queries  
- Displays top matching resumes with similarity scores  

---

## 🧠 How It Works

1. **Resume Upload**  
   Users upload resumes in PDF format  

2. **Text Extraction**  
   Extract text from PDFs  

3. **Embedding Generation**  
   Convert resume text into vector embeddings  

4. **Storage**  
   Store embeddings in vector database (local simulation)  

5. **Query Processing**  
   Convert user query into embedding  

6. **Semantic Matching**  
   Compute cosine similarity between query and resumes  

7. **Results**  
   Return top relevant resumes  

---

## 🛠️ Tech Stack

- **Python**  
- **Streamlit** (UI)  
- **Sentence Transformers** (Embeddings)  
- **NumPy** (Similarity computation)  
- **PyPDF2** (PDF processing)  

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/ai-resume
cd ai-resume
🔍 Example Queries
Python Developer

Machine Learning Engineer

Data Analyst

Backend Developer

🎯 Output
The system returns:

Resume file name

Relevance score (similarity)

Ranked results

📈 Future Enhancements
Integration with real Endee vector database

Persistent storage for embeddings

Resume ranking based on experience and skills

RAG-based chatbot for candidate insights

Cloud deployment with scalability

📌 Conclusion
This project demonstrates how AI-powered semantic search can significantly improve recruitment workflows by enabling faster, more accurate, and intelligent resume filtering.

👩‍💻 Author
Sandeep Kaur


