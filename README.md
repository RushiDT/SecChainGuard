# 🔐 SecChainGuard – IoT Security & Blockchain Access Control Assistant

SecChainGuard is an AI-powered assistant that helps you:

- Perform **STRIDE threat modelling** for IoT systems  
- Design **blockchain-based access control policies** (RBAC/ABAC/capabilities)  
- Explain how **ML-based anomaly detection** fits into the security architecture  
- Use **RAG (Retrieval-Augmented Generation)** to ground answers in curated IoT security knowledge  

The system uses a fine-tuned **TinyLlama 1.1B Chat** model with **LoRA adapters**, served via a **FastAPI backend** and a **Streamlit frontend**.

---

## ✨ Key Features

- 🛡️ **STRIDE Threat Analysis** for IoT devices and networks  
- 🔗 **Blockchain-based Access Control** (Device Registry, Access Policy, Audit Log)  
- 🤖 **ML Anomaly Detection** integration for detecting compromised IoT devices  
- 📚 **RAG-based context injection** from curated IoT security, blockchain, and ML anomaly content  
- 🧪 Custom fine-tuning pipeline using JSONL instruction–response pairs  
- 💻 Web UI using **Streamlit** + **FastAPI** backend API

---

## 🧱 Architecture Overview

The system has three main components:

1. **Frontend (Streamlit)**  
   - Takes user query and IoT system description  
   - Sends request to `/analyze` API  
   - Renders analysis and RAG contexts

2. **Backend (FastAPI)**  
   - `/analyze` endpoint  
   - Combines user input + retrieved context into a structured prompt  
   - Calls the LLM wrapper to generate a 4-section answer  

3. *LLM + RAG Layer* 
   - **LLMWrapper**: loads TinyLlama base model + LoRA adapters  
   - **SimpleRAG**: returns curated IoT/Blockchain/ML anomaly detection snippets  





