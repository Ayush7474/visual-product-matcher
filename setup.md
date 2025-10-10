# 🛠️ Project Setup Guide — Visual Product Matcher

This document tracks the environment and dependency setup for the project.  
It ensures consistency between local development, testing, and deployment.

---

## 📦 1. Python Environment (Backend / AI)
**Purpose:** Used for image processing, embeddings, and API.

**Recommended Python version:** 3.10 or higher  

**Steps (for local setup)**
1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
2. Upgrade Pip:
   ```bash
   pip install --upgrade pip

3.Install dependencies (to be added in requirements.txt later):

torch, transformers, Pillow, faiss-cpu, pandas, fastapi, uvicorn

## 🌐 2. Node.js Environment (Frontend)

Purpose: Used for the React/Vite frontend.

Recommended Node version: LTS (20.x or latest stable)

Steps (for local setup)

Initialize Node project:

npm init -y


Install dependencies (to be added later):

react, react-dom, vite, axios, tailwindcss

Start development server:

npm run dev

🗂️ 3. Folder Structure Plan
visual-product-matcher/
├── backend/             # FastAPI backend
├── frontend/            # React frontend
├── data/                # Dataset & CSV files
├── tools/               # Scripts (e.g., download_images.py)
├── setup.md             # This file
├── README.md
└── .gitignore

🔒 4. Environment Variables (Later)

We will add an .env file for configuration (API keys, paths, etc.).
Make sure .env is listed in .gitignore to avoid committing secrets.

🧾 5. Notes

Follow consistent dependency versions.

Always activate venv before running backend scripts.

Keep all setup steps documented here.
