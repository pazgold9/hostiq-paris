"""
HostIQ Paris - Professional Dashboard
Using Real Review Intelligence Data
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.add_vertical_space import add_vertical_space
import streamlit_antd_components as sac
from streamlit_pills import pills

# ============== PAGE CONFIG ==============
st.set_page_config(
    page_title="HostIQ Paris",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============== CUSTOM CSS ==============
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .main { background: linear-gradient(135deg, #f5f7fa 0%, #e4e8eb 100%); }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown {
        color: white !important;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .glass-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.18);
        margin-bottom: 20px;
    }
    
    .metric-big {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 8px;
    }
    
    .insight-card {
        background: white;
        border-radius: 16px;
        padding: 16px 20px;
        margin: 8px 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        transition: transform 0.2s ease;
        color: #1e293b !important;
    }
    
    .insight-card:hover { transform: translateY(-2px); }
    
    .insight-card span, .insight-card p { color: #1e293b !important; }
    
    .progress-container {
        background: #e2e8f0;
        border-radius: 100px;
        height: 8px;
        overflow: hidden;
    }
    
    .progress-bar {
        height: 100%;
        border-radius: 100px;
        transition: width 0.5s ease;
    }
    
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        color: white;
    }
    
    .property-card {
        background: white;
        border-radius: 16px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border-left: 4px solid #667eea;
    }
    
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 100px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    .badge-success { background: #d1fae5; color: #065f46; }
    .badge-warning { background: #fef3c7; color: #92400e; }
    .badge-danger { background: #fee2e2; color: #991b1b; }
    
    /* Navigation Pills - Much Larger buttons - Multiple selectors */
    [data-testid="stHorizontalBlock"] button,
    .stPills button,
    div[data-baseweb="button-group"] button,
    button[kind="secondary"],
    button[kind="pill"],
    .st-emotion-cache-1inwz65,
    [class*="pills"] button,
    [class*="Pills"] button {
        font-size: 1.4rem !important;
        padding: 18px 40px !important;
        font-weight: 700 !important;
        min-height: 60px !important;
        margin: 0 8px !important;
    }
    
    /* Streamlit pills specific */
    div[data-testid="stHorizontalBlock"] > div > button {
        font-size: 1.4rem !important;
        padding: 18px 40px !important;
        font-weight: 700 !important;
        min-height: 60px !important;
    }
</style>
""", unsafe_allow_html=True)

# ============== CONSTANTS ==============
PRICE_DELTA_25TH = -44.19
PRICE_DELTA_75TH = 27.18

# ============== DATA LOADING (CACHED) ==============
@st.cache_data(ttl=3600)
def load_data():
    # 1. Load property-level data
    properties_df = pd.read_csv('sample_data.csv')
    
    # 2. Load host-level intelligence report
    host_report = pd.read_csv('host_review_intelligence_report.csv')
    
    # 3. Load property review intelligence (source of truth for reviews)
    property_review = pd.read_csv('property_review_intelligence_report.csv')
    
    # Normalize sentiment scores from [-1, 1] to [0, 1]
    def normalize_sentiment(x):
        if pd.isna(x):
            return x
        if x < 0:
            return (x + 1) / 2
        return x
    
    property_review['Avg_Sentiment_Score'] = property_review['Avg_Sentiment_Score'].apply(normalize_sentiment)
    
    # Merge property review data into properties_df
    property_review['Property_ID'] = property_review['Property_ID'].astype(str)
    properties_df['property_id_str'] = properties_df['property_id'].astype(str)
    
    # Update review columns from property_review (source of truth)
    review_mapping = {
        'Total_Reviews': 'review_total',
        'Positive_Reviews': 'review_positive', 
        'Negative_Reviews': 'review_negative',
        'Avg_Sentiment_Score': 'review_sentiment',
        'Cleanliness_Rating': 'rating_cleanliness',
        'Communication_Rating': 'rating_communication',
        'Checkin_Rating': 'rating_checkin',
        'Location_Rating': 'rating_location',
        'Value_Rating': 'rating_value',
        'Advantages': 'review_advantages',
        'Disadvantages': 'review_disadvantages',
        'Suggestions': 'review_suggestions'
    }
    
    for src_col, dst_col in review_mapping.items():
        if src_col in property_review.columns:
            # Create a mapping from property_id to value
            mapping = property_review.set_index('Property_ID')[src_col].to_dict()
            properties_df[dst_col] = properties_df['property_id_str'].map(mapping).combine_first(
                properties_df[dst_col] if dst_col in properties_df.columns else pd.Series()
            )
    
    properties_df = properties_df.drop(columns=['property_id_str'])
    
    # Calculate price delta at property level
    properties_df['price_delta'] = properties_df['prediction'] - properties_df['price']
    
    # Merge property data with host intelligence
    merged = properties_df.merge(
        host_report,
        left_on='seller_id',
        right_on='Host_ID',
        how='inner',
        suffixes=('', '_host')
    )
    
    # Create descriptive listing name from property attributes
    merged['Listing_Name'] = (
        merged['property_type'].str.replace('_', ' ').str.title() + 
        ' in ' + merged['city']
    )
    
    # All data is REAL - from property_review_intelligence_report-2.csv
    # Sentiment Score
    merged['Property_Sentiment'] = merged['review_sentiment']
    
    # Review counts
    merged['Property_Reviews'] = merged['review_total'].fillna(0).astype(int)
    merged['Property_Positive'] = merged['review_positive'].fillna(0).astype(int)
    merged['Property_Negative'] = merged['review_negative'].fillna(0).astype(int)
    
    # Ratings
    merged['Cleanliness_Rating'] = merged['rating_cleanliness']
    merged['Communication_Rating'] = merged['rating_communication']
    merged['Checkin_Rating'] = merged['rating_checkin']
    merged['Location_Rating'] = merged['rating_location']
    merged['Value_Rating'] = merged['rating_value']
    
    # Advantages/Disadvantages/Suggestions
    merged['Property_Advantages'] = merged['review_advantages']
    merged['Property_Disadvantages'] = merged['review_disadvantages']
    merged['Property_Suggestions'] = merged['review_suggestions']
    
    return properties_df, host_report, merged

@st.cache_data(ttl=3600)
def get_hosts_data(_host_report):
    """Use pre-calculated host data from intelligence report"""
    hosts = _host_report[['Host_ID', 'Property_Count', 'Host_Rating', 'Total_Reviews', 
                          'Avg_Sentiment_Score', 'Avg_Price', 'Avg_Predicted_Price', 'Is_Superhost']].copy()
    hosts.columns = ['Host_ID', 'Property_Count', 'Host_Rating', 'Total_Reviews', 
                     'Avg_Sentiment', 'Avg_Price', 'Avg_Predicted', 'Is_Superhost']
    return hosts.sort_values('Property_Count', ascending=False)

# Load all data once
properties_df, host_report, merged_df = load_data()
hosts_with_data = get_hosts_data(host_report)

# ============== SIDEBAR ==============
with st.sidebar:
    st.markdown("## 🏠 HostIQ Paris")
    st.markdown("---")
    
    st.markdown("### Select Host")
    
    selected_host = st.selectbox(
        "Choose a host",
        options=hosts_with_data['Host_ID'].tolist(),
        format_func=lambda x: str(int(x)),
        label_visibility="collapsed"
    )
    
    # Get host data
    host_info = hosts_with_data[hosts_with_data['Host_ID'] == selected_host].iloc[0]
    host_properties = merged_df[merged_df['seller_id'] == selected_host]
    
    # Property Selection (Optional)
    st.markdown("---")
    st.markdown("### 🏠 Property Filter")
    
    # Build property list once
    property_list = host_properties.reset_index(drop=True)
    
    # Create options
    options = ["all"] + [f"prop_{i}" for i in range(len(property_list))]
    
    def format_option(opt):
        if opt == "all":
            return f"📊 All {len(property_list)} Properties"
        idx = int(opt.split("_")[1])
        row = property_list.iloc[idx]
        name = str(row['Listing_Name'])[:25]
        return f"#{idx+1}: {name}... €{row['price']:.0f}"
    
    selected_option = st.selectbox(
        "Select property",
        options=options,
        format_func=format_option,
        label_visibility="collapsed",
        key=f"prop_select_{selected_host}"  # Unique key per host
    )
    
    # Filter data based on selection
    if selected_option == "all":
        single_property_mode = False
        filtered_properties = property_list
    else:
        single_property_mode = True
        idx = int(selected_option.split("_")[1])
        filtered_properties = property_list.iloc[[idx]]
        st.caption(f"🎯 Showing 1 property")

# ============== MAIN CONTENT ==============
if single_property_mode:
    prop_data = filtered_properties.iloc[0]
    prop_name = str(prop_data['Listing_Name'])
    # Check if Superhost
    is_superhost = host_properties['Is_Superhost'].iloc[0] == 1
    superhost_badge = ' • ✨ Superhost' if is_superhost else ''
    
    st.markdown(f"""
    <div class="header-container">
        <h1 style="color: white; margin: 0;">🏠 Single Property Analysis</h1>
        <p style="color: rgba(255,255,255,0.8); margin: 0.5rem 0 0 0; font-size: 1.1rem;">
            {prop_name[:60]}{'...' if len(prop_name) > 60 else ''}
        </p>
        <p style="color: rgba(255,255,255,0.6); margin: 0.3rem 0 0 0; font-size: 0.9rem;">
            Host ID: {selected_host} • Property ID: {prop_data['property_id']}{superhost_badge}
        </p>
    </div>
    """, unsafe_allow_html=True)
else:
    # Check if Superhost
    is_superhost = host_properties['Is_Superhost'].iloc[0] == 1
    superhost_badge = ' • ✨ Superhost' if is_superhost else ''
    
    st.markdown(f"""
    <div class="header-container">
        <h1 style="color: white; margin: 0;">🏠 Portfolio Dashboard</h1>
        <p style="color: rgba(255,255,255,0.8); margin: 0.5rem 0 0 0; font-size: 1.1rem;">
            Host ID: {selected_host}{superhost_badge}
        </p>
    </div>
    """, unsafe_allow_html=True)

# Navigation using sac.segmented for better styling control
selected_tab = sac.segmented(
    items=[
        sac.SegmentedItem(label='📊 Overview'),
        sac.SegmentedItem(label='💰 Revenue'),
        sac.SegmentedItem(label='⭐ Quality & Actions'),
    ],
    align='center',
    size='xl',
    radius='xl',
    use_container_width=False,
)

add_vertical_space(2)

# ============== TAB: DASHBOARD ("מה המצב שלי?") ==============
if selected_tab == "📊 Overview":
    
    # Quick Status Alerts - use Property_Sentiment (real data when available)
    avg_delta = filtered_properties['price_delta'].mean()
    if single_property_mode:
        avg_sentiment = filtered_properties.iloc[0]['Property_Sentiment']
    else:
        # Weighted average by number of reviews for more accurate sentiment
        total_reviews = filtered_properties['Property_Reviews'].sum()
        if total_reviews > 0:
            avg_sentiment = (filtered_properties['Property_Sentiment'] * filtered_properties['Property_Reviews']).sum() / total_reviews
        else:
            avg_sentiment = filtered_properties['Property_Sentiment'].mean()
    if pd.isna(avg_sentiment):
        avg_sentiment = 0.5
    
    # Show quick alerts at the top
    alert_col1, alert_col2 = st.columns(2)
    with alert_col1:
        if avg_delta > PRICE_DELTA_75TH:
            sac.alert(label=f"💰 Revenue: You can increase prices by €{avg_delta:.0f}/night (below market)", color='warning', icon=True)
        elif avg_delta < PRICE_DELTA_25TH:
            sac.alert(label=f"⚠️ Revenue: Your prices are €{abs(avg_delta):.0f}/night above market average", color='error', icon=True)
        else:
            sac.alert(label="✅ Revenue: Your prices match market rates", color='success', icon=True)
    
    with alert_col2:
        # Calculate positive/negative for context
        total_pos = filtered_properties['Property_Positive'].sum()
        total_neg = filtered_properties['Property_Negative'].sum()
        total_reviews = int(total_pos + total_neg)
        
        if avg_sentiment >= 0.8:
            sac.alert(label=f"⭐ Quality: Excellent - {int(total_pos)} positive out of {total_reviews} reviews ({avg_sentiment*100:.0f}%)", color='success', icon=True)
        elif avg_sentiment >= 0.6:
            sac.alert(label=f"👍 Quality: Good - {int(total_pos)} positive, {int(total_neg)} negative out of {total_reviews} reviews", color='warning', icon=True)
        else:
            sac.alert(label=f"📌 Quality: Needs attention - {int(total_neg)} negative out of {total_reviews} reviews ({100-avg_sentiment*100:.0f}% negative)", color='error', icon=True)
    
    add_vertical_space(1)
    
    # KPI Cards - adjust based on single property or portfolio view
    col1, col2, col3, col4 = st.columns(4)
    
    if single_property_mode:
        prop = filtered_properties.iloc[0]
        # Use property-level sentiment (real data)
        prop_sentiment = prop['Property_Sentiment']
        sentiment_display = f"{prop_sentiment*100:.0f}%" if pd.notna(prop_sentiment) else "N/A"
        metrics = [
            ("1", "Property", "#667eea"),
            (f"{prop['host_rating']:.1f}" if pd.notna(prop['host_rating']) else "N/A", "Rating", "#10b981"),
            (f"€{prop['price']:.0f}", "Price/Night", "#f59e0b"),
            (sentiment_display, "Sentiment Score", "#8b5cf6")
        ]
    else:
        # Use calculated weighted sentiment for portfolio
        sentiment_display = f"{avg_sentiment*100:.0f}%" if pd.notna(avg_sentiment) else "N/A"
        metrics = [
            (int(host_info['Property_Count']), "Properties", "#667eea"),
            (f"{host_info['Host_Rating']:.1f}", "Avg Rating", "#10b981"),
            (f"€{host_info['Avg_Price']:.0f}", "Avg Price/Night", "#f59e0b"),
            (sentiment_display, "Sentiment Score", "#8b5cf6")
        ]
    
    for col, (value, label, color) in zip([col1, col2, col3, col4], metrics):
        with col:
            st.markdown(f"""
            <div class="glass-card" style="text-align: center; border-top: 4px solid {color};">
                <p style="font-size: 2.5rem; font-weight: 700; color: {color}; margin: 0;">{value}</p>
                <p class="metric-label">{label}</p>
            </div>
            """, unsafe_allow_html=True)
    
    add_vertical_space(1)
    
    # Map and Properties
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.markdown("### 📍 Property Locations")
        
        # Get price range from ALL host properties (not just filtered) for consistent color scale
        price_min = host_properties['price'].min()
        price_max = host_properties['price'].max()
        
        fig = px.scatter_mapbox(
            filtered_properties,
            lat='lat', lon='long',
            color='price',
            size='guests',
            hover_name='Listing_Name',
            hover_data={'lat': False, 'long': False, 'guests': False, 'price': False, 'Property_Sentiment': False},
            color_continuous_scale='Viridis',
            range_color=[price_min, price_max],  # Keep consistent scale across all views
            zoom=12 if not single_property_mode else 14, height=400
        )
        fig.update_layout(
            mapbox_style="carto-positron",
            margin=dict(l=0, r=0, t=0, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col_right:
        if single_property_mode:
            # For single property - show property details instead of ratings
            st.markdown("### 🏠 Property Details")
            prop = filtered_properties.iloc[0]
            
            details = [
                ("🛏️ Bedrooms", f"{int(prop['bedrooms'])}" if pd.notna(prop['bedrooms']) else "N/A"),
                ("👥 Guests", f"{int(prop['guests'])}"),
                ("🚿 Bathrooms", f"{int(prop['bathrooms'])}" if pd.notna(prop['bathrooms']) else "N/A"),
                ("📊 Reviews", f"{int(prop['Property_Reviews'])}"),
                ("📍 Walk Score", f"{prop['walk_score']:.0f}/100" if pd.notna(prop['walk_score']) else "N/A"),
            ]
            
            for label, value in details:
                st.markdown(f"""
                <div class="insight-card">
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: #374151;">{label}</span>
                        <span style="font-weight: 700; color: #667eea;">{value}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Amenities
            amenities = []
            if prop.get('has_wifi', 0) == 1: amenities.append("WiFi")
            if prop.get('has_ac', 0) == 1: amenities.append("A/C")
            if prop.get('has_kitchen', 0) == 1: amenities.append("Kitchen")
            if prop.get('has_parking', 0) == 1: amenities.append("Parking")
            if prop.get('has_pool', 0) == 1: amenities.append("Pool")
            
            if amenities:
                st.markdown(f"""
                <div class="insight-card" style="border-left: 4px solid #667eea;">
                    <span style="color: #374151;">✨ {', '.join(amenities)}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            # For portfolio - show host-level ratings breakdown
            st.markdown("### 📈 Host Rating Breakdown")
            st.caption("Average across all your properties")
            
            avg_ratings = filtered_properties[['Cleanliness_Rating', 'Communication_Rating', 
                                           'Checkin_Rating', 'Location_Rating', 'Value_Rating']].mean()
            
            categories = ['Cleanliness', 'Communication', 'Check-in', 'Location', 'Value']
            
            for cat, rating in zip(categories, avg_ratings):
                pct = (rating / 5) * 100
                color = '#10b981' if rating >= 4.5 else ('#f59e0b' if rating >= 4.0 else '#ef4444')
                st.markdown(f"""
                <div class="insight-card">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                        <span style="font-weight: 600; color: #374151;">{cat}</span>
                        <span style="color: {color}; font-weight: 700;">{rating:.2f}</span>
                    </div>
                    <div class="progress-container">
                        <div class="progress-bar" style="width: {pct}%; background: {color};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    


# ============== TAB: REVENUE ("איך להרוויח יותר?") ==============
elif selected_tab == "💰 Revenue":
    
    st.markdown("### 💰 How to Increase Revenue?")
    st.caption("AI-powered pricing analysis to maximize your earnings")
    
    add_vertical_space(1)
    
    # Use filtered_properties for calculations
    avg_delta = filtered_properties['price_delta'].mean()
    avg_price = filtered_properties['price'].mean()
    avg_predicted = filtered_properties['prediction'].mean()
    
    # Revenue Opportunity Card
    if avg_delta > PRICE_DELTA_75TH:
        # Calculate potential monthly revenue increase (assuming 20 nights booked)
        monthly_potential = avg_delta * 20
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); padding: 25px; border-radius: 16px; margin-bottom: 20px;">
            <div style="display: flex; align-items: center; gap: 15px;">
                <span style="font-size: 3rem;">💰</span>
                <div>
                    <p style="color: rgba(255,255,255,0.9); font-size: 0.9rem; margin: 0;">REVENUE OPPORTUNITY</p>
                    <h2 style="color: white; margin: 5px 0;">You can raise prices by €{avg_delta:.0f}/night</h2>
                    <p style="color: rgba(255,255,255,0.8); margin: 0;">Potential extra revenue: <strong>€{monthly_potential:.0f}/month</strong> (at 20 nights)</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif avg_delta < PRICE_DELTA_25TH:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); padding: 25px; border-radius: 16px; margin-bottom: 20px;">
            <div style="display: flex; align-items: center; gap: 15px;">
                <span style="font-size: 3rem;">⚠️</span>
                <div>
                    <p style="color: rgba(255,255,255,0.9); font-size: 0.9rem; margin: 0;">PRICING ALERT</p>
                    <h2 style="color: white; margin: 5px 0;">Priced €{abs(avg_delta):.0f}/night above market</h2>
                    <p style="color: rgba(255,255,255,0.8); margin: 0;">Consider lowering prices or adding more value to justify the premium</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 25px; border-radius: 16px; margin-bottom: 20px;">
            <div style="display: flex; align-items: center; gap: 15px;">
                <span style="font-size: 3rem;">✅</span>
                <div>
                    <p style="color: rgba(255,255,255,0.9); font-size: 0.9rem; margin: 0;">OPTIMAL PRICING</p>
                    <h2 style="color: white; margin: 5px 0;">Your prices are well-aligned with the market</h2>
                    <p style="color: rgba(255,255,255,0.8); margin: 0;">Price delta: €{avg_delta:+.0f} - within optimal range</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    add_vertical_space(1)
    
    # Price Cards
    col1, col2, col3 = st.columns(3)
    
    price_label = "Current Price" if single_property_mode else "Current Avg Price"
    predicted_label = "Engine Predicted" if single_property_mode else "Engine Predicted Avg"
    
    with col1:
        st.markdown(f"""
        <div class="glass-card" style="text-align: center;">
            <p style="color: #64748b; font-size: 0.9rem; margin: 0;">{price_label}</p>
            <p style="font-size: 2.5rem; font-weight: 700; color: #1e293b; margin: 0;">€{avg_price:.0f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="glass-card" style="text-align: center;">
            <p style="color: #64748b; font-size: 0.9rem; margin: 0;">{predicted_label}</p>
            <p style="font-size: 2.5rem; font-weight: 700; color: #667eea; margin: 0;">€{avg_predicted:.0f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Positive delta (underpriced) = green, Negative delta (overpriced) = red
        delta_color = '#10b981' if avg_delta > PRICE_DELTA_75TH else ('#ef4444' if avg_delta < PRICE_DELTA_25TH else '#667eea')
        st.markdown(f"""
        <div class="glass-card" style="text-align: center;">
            <p style="color: #64748b; font-size: 0.9rem; margin: 0;">Price Delta</p>
            <p style="font-size: 2.5rem; font-weight: 700; color: {delta_color}; margin: 0;">€{avg_delta:+.0f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Property Pricing Table & Chart - only show for portfolio view (multiple properties)
    if not single_property_mode:
        add_vertical_space(2)
        st.markdown("### 📋 Property Pricing Analysis")
        
        for _, prop in filtered_properties.iterrows():
            delta = prop['price_delta']
            # Positive delta = predicted > actual = can raise price (Underpriced)
            # Negative delta = predicted < actual = should lower price (Overpriced)
            if delta > PRICE_DELTA_75TH:
                status = 'Underpriced'
                status_bg = '#16a34a'  # Green - opportunity to raise price
                status_text = '#ffffff'
            elif delta < PRICE_DELTA_25TH:
                status = 'Overpriced'
                status_bg = '#dc2626'  # Red - should lower price
                status_text = '#ffffff'
            else:
                status = 'Optimal'
                status_bg = '#10b981'
                status_text = '#ffffff'
            
            st.markdown(f"""
            <div style="background: white; padding: 15px 20px; border-radius: 12px; margin: 8px 0; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
                <div style="flex: 2;">
                    <p style="margin: 0; font-weight: 600; color: #1e293b;">{prop['Listing_Name'][:40]}...</p>
                    <p style="margin: 0; font-size: 0.8rem; color: #64748b;">{prop['property_type']}</p>
                </div>
                <div style="text-align: center; flex: 1;">
                    <p style="margin: 0; color: #64748b; font-size: 0.75rem;">Current</p>
                    <p style="margin: 0; font-weight: 700; color: #1e293b;">€{prop['price']:.0f}</p>
                </div>
                <div style="text-align: center; flex: 1;">
                    <p style="margin: 0; color: #64748b; font-size: 0.75rem;">Predicted</p>
                    <p style="margin: 0; font-weight: 700; color: #667eea;">€{prop['prediction']:.0f}</p>
                </div>
                <div style="text-align: center; flex: 1;">
                    <p style="margin: 0; color: #64748b; font-size: 0.75rem;">Delta</p>
                    <p style="margin: 0; font-weight: 700; color: #1e293b;">€{delta:+.0f}</p>
                </div>
                <div style="flex: 1; text-align: right;">
                    <span style="background: {status_bg}; color: {status_text}; padding: 6px 14px; border-radius: 100px; font-size: 0.8rem; font-weight: 600;">{status}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Price vs Predicted chart removed per user request


# ============== TAB: QUALITY & ACTIONS (מאוחד) ==============
elif selected_tab == "⭐ Quality & Actions":
    
    st.markdown("### ⭐ Guest Feedback & Recommended Actions")
    st.caption("What guests say about your properties and what to do next")
    
    add_vertical_space(1)
    
    # Calculate metrics using REAL property-level data
    avg_delta = filtered_properties['price_delta'].mean()
    if single_property_mode:
        avg_sentiment = filtered_properties.iloc[0]['Property_Sentiment']
        if pd.isna(avg_sentiment):
            avg_sentiment = 0.5
    else:
        # Weighted average by number of reviews
        total_reviews = filtered_properties['Property_Reviews'].sum()
        if total_reviews > 0:
            avg_sentiment = (filtered_properties['Property_Sentiment'] * filtered_properties['Property_Reviews']).sum() / total_reviews
        else:
            avg_sentiment = filtered_properties['Property_Sentiment'].mean()
        if pd.isna(avg_sentiment):
            avg_sentiment = 0.5
    
    is_price_issue = avg_delta < PRICE_DELTA_25TH or avg_delta > PRICE_DELTA_75TH
    is_quality_issue = avg_sentiment < 0.7
    
    # ===== SECTION 1: GUEST SENTIMENT + WHAT GUESTS LOVE =====
    add_vertical_space(1)
    st.markdown("### 📊 Guest Sentiment & What They Love")
    
    col_sent, col_advantages = st.columns([1, 2])
    
    with col_sent:
        sentiment_pct = avg_sentiment * 100
        sentiment_color = '#10b981' if avg_sentiment >= 0.8 else ('#f59e0b' if avg_sentiment >= 0.6 else '#ef4444')
        sentiment_label = 'Excellent' if avg_sentiment >= 0.8 else ('Good' if avg_sentiment >= 0.6 else 'Needs Work')
        
        st.markdown(f"""
        <div class="glass-card" style="text-align: center;">
            <div style="position: relative; width: 120px; height: 120px; margin: 0 auto;">
                <svg viewBox="0 0 100 100" style="transform: rotate(-90deg);">
                    <circle cx="50" cy="50" r="45" fill="none" stroke="#e2e8f0" stroke-width="10"/>
                    <circle cx="50" cy="50" r="45" fill="none" stroke="{sentiment_color}" stroke-width="10"
                            stroke-dasharray="{sentiment_pct * 2.83} 283" stroke-linecap="round"/>
                </svg>
                <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); text-align: center;">
                    <p style="font-size: 1.5rem; font-weight: 700; margin: 0; color: {sentiment_color};">{sentiment_pct:.0f}%</p>
                </div>
            </div>
            <p style="color: {sentiment_color}; font-weight: 600; margin-top: 0.5rem; font-size: 0.9rem;">{sentiment_label}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Review counts - all data is real (complete hosts only)
        total_reviews = filtered_properties['Property_Reviews'].sum()
        total_pos = filtered_properties['Property_Positive'].sum()
        total_neg = filtered_properties['Property_Negative'].sum()
        
        st.markdown(f"""
        <div style="display: flex; justify-content: space-around; margin-top: 0.5rem; font-size: 0.85rem;">
            <div style="text-align: center;">
                <p style="font-weight: 700; color: #667eea; margin: 0;">{int(total_reviews)}</p>
                <p style="color: #64748b; margin: 0;">Reviews</p>
            </div>
            <div style="text-align: center;">
                <p style="font-weight: 700; color: #10b981; margin: 0;">{int(total_pos)}</p>
                <p style="color: #64748b; margin: 0;">Positive</p>
            </div>
            <div style="text-align: center;">
                <p style="font-weight: 700; color: #ef4444; margin: 0;">{int(total_neg)}</p>
                <p style="color: #64748b; margin: 0;">Negative</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_advantages:
        from collections import Counter
        
        # ADVANTAGES from REAL property-level review data (What Guests Love)
        st.markdown("#### ✅ What Guests Love")
        all_advantages = []
        # Use Property_Advantages (real data from paris_property_review_intelligence.csv)
        for adv in filtered_properties['Property_Advantages'].dropna():
            all_advantages.extend([a.strip() for a in str(adv).split(';') if a.strip() and 'No issues' not in a])
        
        adv_counts = Counter(all_advantages)
        top_advantages = adv_counts.most_common(6)
        
        if top_advantages:
            adv_cols = st.columns(2)
            for i, (adv, count) in enumerate(top_advantages):
                with adv_cols[i % 2]:
                    st.markdown(f"""
                    <div class="insight-card" style="border-left: 4px solid #10b981; padding: 8px 12px;">
                        <span style="color: #10b981;">👍</span> <span style="color: #374151; font-size: 0.85rem;">{adv}</span>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No specific advantages from reviews yet.")
    
    # ===== SECTION 3: ISSUES IDENTIFIED (Real Disadvantages from Reviews) =====
    add_vertical_space(2)
    st.markdown("### ⚠️ Areas for Improvement")
    st.caption("Issues identified from real guest reviews")
    
    # Collect REAL disadvantages from property-level reviews
    all_disadvantages = []
    for dis in filtered_properties['Property_Disadvantages'].dropna():
        dis_str = str(dis)
        if 'No issues' not in dis_str and dis_str.strip():
            all_disadvantages.extend([d.strip() for d in dis_str.split(';') if d.strip() and 'No issues' not in d])
    
    dis_counts = Counter(all_disadvantages)
    top_disadvantages = dis_counts.most_common(6)
    
    if top_disadvantages:
        dis_cols = st.columns(2)
        for i, (dis, count) in enumerate(top_disadvantages):
            with dis_cols[i % 2]:
                st.markdown(f"""
                <div class="insight-card" style="border-left: 4px solid #ef4444; padding: 8px 12px;">
                    <span style="color: #ef4444;">⚠️</span> <span style="color: #374151; font-size: 0.85rem;">{dis}</span>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.success("🎉 No issues identified - guests are happy!")
    
    # ===== SECTION 4: AI SUGGESTIONS (Solutions based on real issues) =====
    add_vertical_space(2)
    st.markdown("### 💡 Recommendations")
    st.caption("Actionable solutions based on guest feedback")
    
    # Collect REAL suggestions from property-level data (Property_Suggestions)
    unique_suggestions = set()
    for sug in filtered_properties['Property_Suggestions'].dropna():
        sug_str = str(sug)
        if 'great work' in sug_str.lower():
            continue
        # Split by || to get individual suggestions
        parts = sug_str.split('||')
        for part in parts:
            part = part.strip()
            if part and 'No specific' not in part:
                unique_suggestions.add(part)
    
    if unique_suggestions:
        for sug_clean in list(unique_suggestions):
            # Try to extract category from suggestion (e.g., "[Pricing]:", "[Quality]:")
            if sug_clean.startswith('[') and ']:' in sug_clean:
                category = sug_clean[1:sug_clean.index(']')]
                suggestion_text = sug_clean[sug_clean.index(']:')+2:].strip()
                
                # Color and icon based on category
                if 'Pricing' in category or 'Price' in category:
                    icon = "💰"
                    color = "#3b82f6"
                elif 'Cleanliness' in category or 'Clean' in category:
                    icon = "🧹"
                    color = "#8b5cf6"
                elif 'Quality' in category:
                    icon = "⭐"
                    color = "#f59e0b"
                elif 'Amenities' in category:
                    icon = "🛋️"
                    color = "#10b981"
                elif 'Communication' in category:
                    icon = "💬"
                    color = "#ec4899"
                elif 'Noise' in category:
                    icon = "🔇"
                    color = "#6366f1"
                elif 'Dirty' in category or 'Bathroom' in category:
                    icon = "🧹"
                    color = "#8b5cf6"
                elif 'Too Small' in category or 'Too Hot' in category:
                    icon = "🏠"
                    color = "#f97316"
                else:
                    icon = "💡"
                    color = "#667eea"
            else:
                category = "Suggestion"
                suggestion_text = sug_clean
                icon = "💡"
                color = "#667eea"
            
            st.markdown(f"""
            <div style="background: white; border-radius: 12px; padding: 16px 20px; margin: 10px 0; 
                        border-left: 4px solid {color}; box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
                <div style="display: flex; align-items: flex-start; gap: 12px;">
                    <span style="font-size: 1.5rem;">{icon}</span>
                    <div>
                        <p style="font-size: 0.75rem; color: {color}; text-transform: uppercase; letter-spacing: 1px; margin: 0; font-weight: 600;">{category}</p>
                        <p style="color: #374151; margin: 5px 0 0 0; line-height: 1.5;">{suggestion_text}</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("🎉 No improvements needed - Keep up the excellent work!")


# ============== FOOTER ==============
add_vertical_space(3)
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #94a3b8; font-size: 0.85rem; padding: 1rem;">
    🏠 HostIQ Paris Dashboard • Powered by Real Review Intelligence Data
</div>
""", unsafe_allow_html=True)
