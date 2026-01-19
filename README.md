# 📄 AI-Based Document Retrieval & Q&A System

## 🧩 Problem Statement

Users often work with large PDF documents such as research papers, legal files, reports, and manuals.  
Finding specific information inside these documents is time-consuming and inefficient because traditional search tools rely only on keyword matching and do not understand context.

There is a need for an intelligent system that can:
- Understand document content
- Answer user questions in natural language
- Provide accurate, context-based responses

---

## 💡 Solution Overview

This project is an **AI-Based Document Retrieval and Question Answering System** that allows users to upload PDF documents and ask questions related to the document.  
The system uses **AI and Natural Language Processing (NLP)** to understand the document and return precise answers based only on the document content.

---

## 🤖 What is AI-Based Document Retrieval?

AI-Based Document Retrieval uses machine learning models to understand the meaning of text instead of searching for exact keywords.  
It converts document text into **vector embeddings**, enabling semantic search and intelligent question answering.

---

## ⚙️ How the System Works

1. User uploads a PDF document  
2. Text is extracted from the PDF  
3. Text is split into smaller chunks  
4. Each chunk is converted into vector embeddings  
5. Embeddings are stored in a vector database  
6. User asks a question  
7. Relevant document sections are retrieved  
8. AI model generates an answer using document context only  

---

## 🛠️ Technology Stack

- **Frontend:** Streamlit  
- **LLM:** Meta LLaMA 3.2 (1B Instruct)  
- **Embeddings:** Sentence Transformers (MiniLM)  
- **Vector Database:** ChromaDB  
- **Framework:** LangChain  
- **PDF Processing:** PyPDF2  

---

## ✨ Features

- Upload PDF documents  
- Chat-based question answering  
- Context-aware responses  
- Prevents AI hallucination  
- Simple and interactive UI  

---

## 🎯 Use Cases

- Student study and exam preparation  
- Legal and policy document analysis  
- Research paper understanding  
- Corporate document review  

---

## 🚀 Future Enhancements

- OCR support for scanned PDFs  
- Multi-document support  
- Answer citation with page numbers  
- Cloud deployment  

---

## 👥 Team Details

**Team Name:** Celestial Coders  
**Project Type:** AI / NLP / LLM-Based Application  

---

## 📌 How to Run the Project

1. Clone the repository  
2. Install required dependencies  
3. Add your Hugging Face API token in `.env`  
4. Run the Streamlit application  

```bash
streamlit run app.py

```

---


## 🖼️ Application Screenshot

<img width="1856" height="883" alt="Screenshot 2026-01-14 220621" src="https://github.com/user-attachments/assets/067fa5fb-a2da-4f58-b9d8-01e372c33ee6" />

---

<img width="1829" height="907" alt="Screenshot 2026-01-14 220731" src="https://github.com/user-attachments/assets/6f3b1a06-115e-4574-91ec-3b3395718bc0" />

