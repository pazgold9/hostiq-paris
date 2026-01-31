# HostIQ - Airbnb Host Intelligence Platform

A data driven dashboard for Airbnb hosts in Paris, providing pricing optimization and guest feedback analysis.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Overview

HostIQ Paris helps Airbnb hosts optimize their listings through:

- **Smart Pricing Engine** - ML-based price predictions using Random Forest
- **Guest Feedback Intelligence** - NLP-powered review analysis with sentiment scoring
- **Portfolio Dashboard** - Visual analytics for multi-property hosts

## Features

### Dashboard Views
1. **Overview** - Key metrics, alerts, and property map
2. **Revenue** - Price optimization recommendations
3. **Quality & Actions** - Review insights and improvement suggestions

### Key Capabilities
- Price vs. predicted price comparison
- Sentiment analysis of guest reviews
- Automated identification of property advantages/disadvantages
- Actionable recommendations based on review patterns

## Project Structure

```
hostiq_app/
├── app.py                          # Streamlit application
├── property_data.csv               # Property-level data (one row per property)
├── host_data.csv                   # Host-level aggregated data
├── requirements.txt                # Python dependencies
├── render.yaml                     # Render deployment config
└── notebooks/
    ├── data_preparation.ipynb      # Data preprocessing pipeline
    ├── Smart Pricing Engine.ipynb  # ML pricing model
    └── Guest Feedback Intelligence System(paris_data).ipynb  # NLP analysis
```

## Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/pazgold9/hostiq-paris.git
   cd hostiq-paris/hostiq_app
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

### Local Development

```bash
cd hostiq_app
streamlit run app.py
```

The app will be available at `http://localhost:8501`

### Production Deployment (Render)

The app is configured for deployment on Render. Simply connect your GitHub repository and Render will automatically deploy using `render.yaml`.

**Live Demo:** [https://hostiq-paris.onrender.com](https://hostiq-paris.onrender.com)

