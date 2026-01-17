"""
🏠 HostIQ Dashboard - Databricks Streamlit App
With Azure Blob Storage Connection
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
import io

# ============================================================
# PAGE CONFIG - Must be first Streamlit command
# ============================================================

st.set_page_config(
    page_title="HostIQ Dashboard",
    page_icon="🏠",
    layout="wide"
)

# ============================================================
# AZURE CONFIGURATION
# ============================================================

AZURE_CONFIG = {
    "storage_account": "lab94290",
    "container": "airbnb",
    "sas_token": "sp=rle&st=2025-12-24T17:37:04Z&se=2026-02-28T01:52:04Z&spr=https&sv=2024-11-04&sr=c&sig=a0lx%2BS6PuS%2FvJ9Tbt4NKdCJHLE9d1Y1D6vpE1WKFQtk%3D",
    "parquet_file": "airbnb_1_12_parquet/part-00000-tid-1637687860512127859-6b9a9b3d-9a1b-4463-b4a9-bd5a0f7f348f-91-1-c000.snappy.parquet"
}

# ============================================================
# ALLOWED LOCATIONS (for filtering)
# ============================================================

ALLOWED_LOCATIONS = {
    "Argentina": ["Buenos Aires", "Córdoba", "Mar del Plata"],
    "Australia": ["Melbourne", "Sydney", "Brisbane"],
    "Brazil": ["Rio de Janeiro", "São Paulo", "Salvador"],
    "Canada": ["Toronto", "Vancouver", "Montreal"],
    "France": ["Paris", "Nice", "Lyon"],
    "Germany": ["Berlin", "Munich", "Hamburg"],
    "Italy": ["Rome", "Milan", "Florence"],
    "Mexico": ["Mexico City", "Cancún", "Playa del Carmen"],
    "Spain": ["Barcelona", "Madrid", "Valencia"],
    "United Kingdom": ["London", "Edinburgh", "Manchester"],
    "United States": ["New York", "Los Angeles", "Miami"]
}

ALL_COUNTRIES = list(ALLOWED_LOCATIONS.keys())

# Build city to country mapping
CITY_TO_COUNTRY = {}
ALL_CITIES = []
for country, cities in ALLOWED_LOCATIONS.items():
    ALL_CITIES.extend(cities)
    for city in cities:
        CITY_TO_COUNTRY[city] = country

# ============================================================
# DATA LOADING FUNCTIONS
# ============================================================

def load_sample_data():
    """Generate sample data for demo/fallback"""
    np.random.seed(42)
    n = 5000
    
    city_choices = np.random.choice(ALL_CITIES, n)
    
    df = pd.DataFrame({
        'name': [f'Beautiful Property {i}' for i in range(n)],
        'price': np.random.randint(50, 500, n),
        'rating': np.round(np.random.uniform(3.5, 5.0, n), 2),
        'reviews_count': np.random.randint(5, 300, n),
        'category': np.random.choice(['Stays', 'Experiences'], n, p=[0.9, 0.1]),
        'city': city_choices,
        'country': [CITY_TO_COUNTRY[c] for c in city_choices],
        'seller_id': np.random.choice([f'host_{i}' for i in range(500)], n)
    })
    
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_azure_data():
    """
    Load data directly from Azure Blob Storage using HTTP request.
    No Spark needed - works in any Python environment!
    """
    storage_account = AZURE_CONFIG["storage_account"]
    container = AZURE_CONFIG["container"]
    sas_token = AZURE_CONFIG["sas_token"]
    parquet_file = AZURE_CONFIG["parquet_file"]
    
    # Build the URL for direct HTTP access
    url = f"https://{storage_account}.blob.core.windows.net/{container}/{parquet_file}?{sas_token}"
    
    try:
        # Download the parquet file
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        # Read into pandas
        df = pd.read_parquet(io.BytesIO(response.content))
        
        return df, "Azure Blob Storage"
        
    except requests.exceptions.RequestException as e:
        st.warning(f"⚠️ Could not connect to Azure: {str(e)[:100]}")
        return None, None
    except Exception as e:
        st.warning(f"⚠️ Error reading data: {str(e)[:100]}")
        return None, None


@st.cache_data(ttl=3600)
def load_data():
    """Main data loading function - tries Azure first, falls back to sample"""
    
    # Try Azure first
    df, source = load_azure_data()
    
    if df is not None and len(df) > 0:
        # Standardize column names (lowercase)
        df.columns = df.columns.str.lower()
        
        # Add country column if city exists
        if 'city' in df.columns:
            df['country'] = df['city'].map(CITY_TO_COUNTRY)
            # Filter to only allowed cities
            df = df[df['country'].notna()]
        
        # Ensure required columns exist
        if 'price' not in df.columns and 'price_usd' in df.columns:
            df['price'] = df['price_usd']
        
        if 'reviews_count' not in df.columns and 'number_of_reviews' in df.columns:
            df['reviews_count'] = df['number_of_reviews']
            
        return df, source
    
    # Fallback to sample data
    return load_sample_data(), "Demo Data"


# ============================================================
# MAIN APP
# ============================================================

# Title
st.title("🏠 HostIQ Dashboard")
st.markdown("**Airbnb Portfolio Analytics Platform**")

# Load data with spinner
with st.spinner("📥 Loading data from Azure..."):
    df, data_source = load_data()

# Show data source
if data_source == "Azure Blob Storage":
    st.success(f"✅ Loaded {len(df):,} listings from **{data_source}**")
else:
    st.info(f"ℹ️ Using {data_source} ({len(df):,} listings)")

# Sidebar filters
st.sidebar.header("🔍 Filters")

# Data source indicator
st.sidebar.markdown(f"📊 **Source:** {data_source}")
st.sidebar.markdown("---")

# Country filter
selected_country = st.sidebar.selectbox("🌍 Country", ["All"] + ALL_COUNTRIES)

# City filter (dynamic based on country)
if selected_country != "All":
    available_cities = ALLOWED_LOCATIONS.get(selected_country, [])
else:
    available_cities = ALL_CITIES

selected_city = st.sidebar.selectbox("🏙️ City", ["All"] + sorted(available_cities))

# Host filter (if seller_id exists)
if 'seller_id' in df.columns:
    unique_hosts = df['seller_id'].dropna().unique().tolist()[:30]
    hosts = ["All"] + sorted([str(h) for h in unique_hosts])
    selected_host = st.sidebar.selectbox("🏠 Host", hosts)
else:
    selected_host = "All"

# Apply filters
df_filtered = df.copy()

if selected_country != "All" and 'country' in df_filtered.columns:
    df_filtered = df_filtered[df_filtered['country'] == selected_country]

if selected_city != "All" and 'city' in df_filtered.columns:
    df_filtered = df_filtered[df_filtered['city'] == selected_city]

if selected_host != "All" and 'seller_id' in df_filtered.columns:
    df_filtered = df_filtered[df_filtered['seller_id'].astype(str) == selected_host]

st.sidebar.markdown("---")
st.sidebar.metric("📊 Showing", f"{len(df_filtered):,} listings")

# KPIs
st.markdown("---")
st.markdown("### 📊 Key Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🏠 Properties", f"{len(df_filtered):,}")

with col2:
    avg_price = df_filtered['price'].mean() if 'price' in df_filtered.columns and len(df_filtered) > 0 else 0
    st.metric("💰 Avg Price", f"${avg_price:.0f}")

with col3:
    avg_rating = df_filtered['rating'].mean() if 'rating' in df_filtered.columns and len(df_filtered) > 0 else 0
    st.metric("⭐ Avg Rating", f"{avg_rating:.2f}")

with col4:
    if 'reviews_count' in df_filtered.columns and len(df_filtered) > 0:
        total_reviews = int(df_filtered['reviews_count'].sum())
    else:
        total_reviews = 0
    st.metric("💬 Reviews", f"{total_reviews:,}")

# Charts
st.markdown("---")
st.markdown("### 📈 Analytics")

col1, col2 = st.columns(2)

with col1:
    if 'price' in df_filtered.columns and len(df_filtered) > 0:
        fig = px.histogram(
            df_filtered, x='price', nbins=25,
            title='💰 Price Distribution',
            color_discrete_sequence=['#6366f1']
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No price data to display")

with col2:
    if 'city' in df_filtered.columns and len(df_filtered) > 0:
        city_counts = df_filtered['city'].value_counts().head(10)
        fig = px.pie(
            values=city_counts.values,
            names=city_counts.index,
            title='🌍 Top Cities',
            hole=0.4
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No city data to display")

# Rating vs Price scatter
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    if 'price' in df_filtered.columns and 'rating' in df_filtered.columns and len(df_filtered) > 0:
        sample_size = min(500, len(df_filtered))
        fig = px.scatter(
            df_filtered.sample(sample_size),
            x='price', y='rating',
            color='country' if 'country' in df_filtered.columns else None,
            title='⭐ Rating vs Price',
            opacity=0.6
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

with col2:
    if 'country' in df_filtered.columns and len(df_filtered) > 0:
        agg_dict = {'price': 'mean'} if 'price' in df_filtered.columns else {}
        if 'rating' in df_filtered.columns:
            agg_dict['rating'] = 'mean'
        
        country_stats = df_filtered.groupby('country').size().reset_index(name='count')
        
        if 'price' in df_filtered.columns:
            price_stats = df_filtered.groupby('country')['price'].mean().reset_index()
            country_stats = country_stats.merge(price_stats, on='country')
        
        fig = px.bar(
            country_stats.sort_values('count', ascending=True).tail(10),
            x='count', y='country',
            orientation='h',
            title='🌍 Listings by Country',
            color='price' if 'price' in country_stats.columns else None,
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

# Table
st.markdown("---")
st.markdown("### 🏠 Properties")

if len(df_filtered) > 0:
    # Select available columns
    possible_cols = ['name', 'city', 'country', 'price', 'rating', 'reviews_count', 'category']
    display_cols = [c for c in possible_cols if c in df_filtered.columns]
    
    if display_cols:
        st.dataframe(
            df_filtered[display_cols].head(50),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.dataframe(df_filtered.head(50), use_container_width=True, hide_index=True)
else:
    st.info("No properties match your filters")

# Footer
st.markdown("---")
st.markdown(
    f"<center style='color: gray;'>🏠 HostIQ Dashboard | Data: {data_source} | Powered by Databricks</center>",
    unsafe_allow_html=True
)
