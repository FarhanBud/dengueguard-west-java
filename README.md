# DengueGuard-Agent: West Java Dengue Mitigation Decision Support System

Decision-support web application built with **Streamlit** for municipal dengue risk analysis and Human-in-the-Loop (HITL) intervention governance across 27 regencies and cities in West Java (2016–2025 historical dataset and 2026+ baseline projections), grounded in the Indonesian Ministry of Health (*Pedoman DBD Kemenkes RI 2017*) guidelines via ChromaDB RAG and Google Gemini.

## Key Features
- **Landing Introduction & Glossary:** Operational guide and terminology definitions (`IR`, `CFR`, `DIBI BNPB`, `PE & Fogging Fokus`, `MCP Tools`, `RAG`, `HITL`, `SHA-256`).
- **Role-Based Workspaces:** Separate portals for **Analis Dinas Kesehatan (P2P)** (investigation, drafting, dispatch & execute) and **Kepala Daerah (Bupati / Walikota)** (executive `Approve` / `Reject` gate).
- **2-Layer Topic Guardrail:** Supports natural public-health field queries while blocking prompt injections and off-topic requests prior to LLM invocation.
- **Tamper-Evident Audit Ledger:** SHA-256 hash-chained activity ledger (`audit_log.jsonl`).

## Deployment on Streamlit Community Cloud
1. Push the core application files (`streamlit_app.py`, `requirements.txt`, `.streamlit/config.toml`, `.gitignore`, `west_java_dengue_master_2016_2025.csv`, `Pedoman_DBD_Kemenkes_2017.pdf`, `dengue_logo.png`) to a GitHub repository.
2. Open [share.streamlit.io](https://share.streamlit.io) and click **Create app**.
3. Select the repository and set **Main file path** to `streamlit_app.py`.
4. Under **Advanced settings -> Secrets** (or **Settings -> Secrets**), add:
   ```toml
   GEMINI_API_KEY = "your_google_ai_studio_api_key"
   ```
   If no API key is configured, the platform runs safely in deterministic Rule-Based Fallback mode.

---
Developed by **Muhamad Farhan Budiana** & **Nadya Ayu Lestari** — Department of Electrical Engineering, Universitas Indonesia. All Rights Reserved.
