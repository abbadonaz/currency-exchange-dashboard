# 💱 Currency Exchange Dashboard

An educational machine learning project that explores anomaly detection in currency exchange rates using algorithms such as **Isolation Forest**.

This repository includes both the **dashboard implementation** and links to my **learning notes**, which document the process.
<img width="1875" height="671" alt="image" src="https://github.com/user-attachments/assets/604f7860-5d34-4b0b-8aa8-21815083d952" />

---

## 📊 Project Overview

- Built with: Python, Streamlit, Scikit-learn, Pandas  
- Core algorithms: Isolation Forest (for anomaly detection in exchange rate data)  
- Purpose: Educational — combining practical coding with structured ML learning.  
---

## 📝 Learning Notes

As part of this project, I documented my learning journey using **NotebookLM**.  
These notes explain key machine learning algorithms and their applications.

- 📘 [Isolation Forest Explaination](./Notes/IsolationForest.md)  

---

## ⚠️ Disclaimer

Parts of the learning notes were **generated with the assistance of NotebookLM** and then curated by me.  
They are included here to **showcase my personal learning process** — not as finalized research material.  

---

## 🚀 Getting Started

Clone the repository:

```bash
git clone https://github.com/abbadonaz/currency-exchange-dashboard.git
cd currency-exchange-dashboard
```

## ⚡ Quickstart

**Prerequisites:** Python 3.11+ and Git installed.

1. Create and activate a virtual environment (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

(Or on Windows CMD: `\.venv\Scripts\activate`, macOS/Linux: `source .venv/bin/activate`)

2. Install dependencies:

```bash
pip install -r requirements.txt
# (Optional) install the DescriptiveAnalytics extras:
# pip install -r DescriptiveAnalytics/requirements.txt
```

3. (Optional) Create a `.env` file at the project root to override defaults (see `src/config.py`). Example:

```env
BASE_CURRENCY=EUR
CACHE_TTL_MIN=60
DATA_SOURCE=ECB
```

4. Run the app:

```bash
streamlit run main.py
```

Then open http://localhost:8501 in your browser. To run on a different port, add `--server.port <PORT>` to the command.

---
