# ⚡ SupportSync — Google Cloud Vertex AI & BigQuery Support Intelligence

[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Google Cloud](https://img.shields.io/badge/GCP-Vertex%20AI%20%7C%20BigQuery-4285F4?style=for-the-badge&logo=googlecloud&logoColor=white)](https://cloud.google.com/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Hackathon](https://img.shields.io/badge/Project-Google%20Cloud%20Hackathon-00C853?style=for-the-badge)](README.md)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**SupportSync** is an enterprise customer support intelligence platform engineered for the **Google Cloud AI Hackathon**. Built with **Streamlit**, **Google Cloud BigQuery**, and **Google Cloud Vertex AI**, SupportSync ingests customer support tickets, scores customer sentiment and urgency, queries historical resolution databases, and automatically drafts context-aware responses for support agents.

---

## 📌 Architecture & GCP Data Flow

```
[ Inbound Customer Support Ticket ]
                  |
                  v
+-------------------------------------------------------------+
|                 Streamlit SupportSync Portal                |
|                      (`supportsync.py`)                     |
+-------------------------------------------------------------+
                  |
                  +-----------------------------+
                  |                             |
                  v                             v
+-------------------------------+ +-------------------------------+
|     GCP Vertex AI Endpoint    | |     Google Cloud BigQuery     |
|   Sentiment & Urgency Scoring | |   Historical Ticket Indexing  |
|   & Resolution Text Synthesis | |   & SLA Analytics Store       |
+-------------------------------+ +-------------------------------+
                  |                             |
                  +--------------+--------------+
                                 |
                                 v
+-------------------------------------------------------------+
|        Unified Agent Dashboard with Auto-Drafted Reply      |
+-------------------------------------------------------------+
```

---

## ✨ Features

* 📊 **Automated Sentiment Analysis:** Evaluates customer frustration, urgency, and churn risk on ticket ingestion.
* 🔍 **BigQuery Vector & Ticket Lookup:** Connects to BigQuery datasets to retrieve historical resolutions to similar customer issues.
* ✍️ **Agent Response Generator:** Synthesizes professional, empathetic resolution replies adhering to enterprise SLAs.

---

## 📁 Repository Structure

```text
supportsync-hackathon/
├── supportsync.py      # Streamlit application, GCP BigQuery & Vertex AI client
├── LICENSE             # MIT License
└── README.md
```

---

## 🚀 Getting Started

### 1. Prerequisites
* Python 3.10+
* Google Cloud Platform account with Vertex AI and BigQuery APIs enabled
* Google Cloud SDK (`gcloud auth application-default login`)

### 2. Installation
```bash
git clone https://github.com/Tarunjit45/supportsync-hackathon.git
cd supportsync-hackathon

pip install streamlit google-cloud-bigquery google-cloud-aiplatform requests
```

### 3. Run SupportSync
```bash
streamlit run supportsync.py
```

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
