# HostIQ - Airbnb Host Intelligence Platform

A data driven platform for Airbnb hosts, providing ML powered pricing optimization and NLP based guest feedback analysis.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## Part 1: The Dashboard App

The Streamlit dashboard provides visual analytics for Airbnb hosts, running on **sample data of 209 hosts based in Paris**.

### Features

- **Overview** - Key metrics, property map, and host ratings breakdown
- **Revenue** - Price optimization with ML predictions vs. actual prices
- **Quality & Actions** - Guest sentiment analysis and improvement recommendations

### Live Demo

**[https://hostiq-paris.onrender.com](https://hostiq-paris.onrender.com)**

### Run Locally

```bash
# Clone and navigate
git clone https://github.com/pazgold9/hostiq-paris.git
cd hostiq-paris/hostiq_app

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies and run
pip install -r requirements.txt
streamlit run app.py
```

The app will be available at `http://localhost:8501`

---

## Part 2: Property Analysis Notebook

The **Select Property ID.ipynb** notebook allows you to analyze any individual property from the dataset.

### What It Does

1. **Price Prediction** - Enter a property ID and get the market fair price according to our Smart Pricing Engine
2. **Review Intelligence Report** - Get a detailed analysis including:
   - Sentiment analysis of all reviews (using DistilBERT)
   - Category ratings (Cleanliness, Communication, Check-in, Location, Value)
   - Extracted advantages and disadvantages with frequency counts
   - Actionable suggestions sourced from Airbnb's knowledge base

### Usage

1. Open `Select Property ID.ipynb`
2. Set your property ID in the `PROPERTY_ID` variable
3. Run all cells to get the price prediction and review intelligence report

---

## Part 3: Training Notebooks (Databricks)

The notebooks in `notebooks/` contain the ML and NLP pipelines used to generate the model and processed data. These are designed to run on **Databricks** and will take a significant amount of time to complete.

| Notebook | Purpose |
|----------|---------|
| `Smart Pricing Engine.ipynb` | Training for price prediction |
| `Guest Feedback Intelligence System (graphs).ipynb` | Training NLP sentiment analysis and visualization |
| `poi_data_collection.ipynb` | Points of Interest data collection for location features |
| `collect_Listing_Quality_Knowledge_Base.py` | scraping browser to collect Airbnb help articles for the suggestion system. |

> **Note:** The scripts contain **only the scraping code**, not the scraped data. The script demonstrates how to use Playwright with Bright Data's proxy to access Airbnb's help center articles.

