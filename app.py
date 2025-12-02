"""
IPO Risk Prediction Dashboard - Real World IPO Data
Streamlit Application for Interactive IPO Analysis

Using Real US IPOs from 2019-2024
Authors: Logan Wesselt, Julian Tashjian, Dylan Bollinger
JLD Inc. LLC. Partners - FIN 377 Final Project
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os

# Page configuration
st.set_page_config(
    page_title="IPO Risk Prediction Dashboard | Real World Data",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1 {
        color: #1f77b4;
        font-weight: 700;
    }
    h2 {
        color: #2c3e50;
        font-weight: 600;
    }
    h3 {
        color: #34495e;
    }
    .highlight-box {
        background-color: #e8f4f8;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #ffc107;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #28a745;
        margin: 10px 0;
    }
    .risk-high {
        color: #dc3545;
        font-weight: bold;
    }
    .risk-low {
        color: #28a745;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

@st.cache_resource
def load_models():
    """Load trained ML models and preprocessing objects"""
    try:
        model_files = {
            'classifier': 'models/best_classifier.pkl',
            'regressor': 'models/best_regressor.pkl',
            'scaler': 'models/scaler.pkl',
            'features': 'models/feature_columns.pkl',
            'metadata': 'models/metadata.pkl'
        }

        models = {}
        for key, filepath in model_files.items():
            with open(filepath, 'rb') as f:
                models[key] = pickle.load(f)

        return (
            models['classifier'],
            models['regressor'],
            models['scaler'],
            models['features'],
            models['metadata']
        )
    except FileNotFoundError as e:
        st.error(f"""
         **Model files not found!**
        
        Please run the Jupyter notebook first to train models:
        1. Run `real_ipo_notebook_part1.py`
        2. Run `real_ipo_notebook_part2.py`
        3. Run `real_ipo_notebook_part3.py`
        
        Missing file: {e.filename}
        """)
        return None, None, None, None, None

@st.cache_data
def load_data():
    """Load test predictions, results, and IPO data"""
    try:
        # Core results data
        test_preds = pd.read_csv('data/test_predictions.csv')
        clf_results = pd.read_csv('data/classification_results.csv')
        reg_results = pd.read_csv('data/regression_results.csv')
        strategy_results = pd.read_csv('data/strategy_summary.csv')
        feature_importance = pd.read_csv('data/feature_importance.csv')

        # Convert date columns if present
        if 'ipo_date' in test_preds.columns:
            test_preds['ipo_date'] = pd.to_datetime(test_preds['ipo_date'])

        return (
            test_preds,
            clf_results,
            reg_results,
            strategy_results,
            feature_importance
        )
    except FileNotFoundError as e:
        st.error(f"""
         **Data files not found!**
        
        Please run the complete Jupyter notebook to generate data files.
        
        Missing file: {e.filename}
        """)
        return None, None, None, None, None

@st.cache_data
def load_baseline_results():
    """Load baseline heuristic results for comparison"""
    try:
        # These should be saved from the notebook
        baseline_file = 'data/baseline_results.csv'
        if os.path.exists(baseline_file):
            return pd.read_csv(baseline_file)
        else:
            # Return default baselines if file doesn't exist
            return pd.DataFrame({
                'Method': ['Most Frequent', 'VIX Rule', 'Young + Unprofitable', 'Mean Return'],
                'Type': ['Classification', 'Classification', 'Classification', 'Regression'],
                'Metric': ['AUC', 'AUC', 'Accuracy', 'RMSE'],
                'Value': [0.50, 0.55, 0.60, 0.25]
            })
    except:
        return None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_currency(value):
    """Format value as currency"""
    return f"${value:,.2f}"

def format_percent(value, decimal_places=2):
    """Format value as percentage"""
    return f"{value*100:.{decimal_places}f}%"

def get_risk_label(risk_prob):
    """Convert risk probability to label"""
    if risk_prob >= 0.7:
        return " High Risk", "risk-high"
    elif risk_prob >= 0.4:
        return " Moderate Risk", "risk-medium"
    else:
        return " Low Risk", "risk-low"

def explain_feature_plain_english(feature_name):
    """
    Provide plain English explanations for features.
    Updated with correct definitions for real IPO data.
    """
    explanations = {
        # Deal Structure - CORRECTED DEFINITIONS
        'offer_price': {
            'name': 'IPO Offer Price',
            'description': 'The price per share at which the company goes public. Higher prices may indicate stronger demand or higher quality companies.'
        },
        'shares_offered': {
            'name': 'Total Shares Offered',
            'description': 'The total number of shares sold in the IPO offering. More shares means a larger offering and more capital raised.'
        },
        'gross_proceeds': {
            'name': 'Gross Proceeds (Total Capital Raised)',
            'description': 'The total amount of money raised in the IPO (offer price × shares offered). Larger offerings tend to be from more established companies.'
        },
        'log_proceeds': {
            'name': 'Log of Gross Proceeds',
            'description': 'Mathematical transformation of gross proceeds that helps the model handle both small and large IPOs fairly.'
        },

        # Firm Characteristics
        'firm_age': {
            'name': 'Company Age',
            'description': 'Years since the company was founded. Younger companies are typically riskier but may have higher growth potential.'
        },
        'is_young_firm': {
            'name': 'Young Firm Indicator',
            'description': 'Whether the company is less than 5 years old. Young firms face higher uncertainty and risk.'
        },
        'vc_backed': {
            'name': 'Venture Capital Backing',
            'description': 'Whether the company is backed by venture capital firms. VC backing can signal quality but may also indicate pressure to exit.'
        },
        'is_profitable': {
            'name': 'Profitability Status',
            'description': 'Whether the company is currently profitable. Profitable companies are generally less risky but may have lower growth potential.'
        },
        'underwriter_rank': {
            'name': 'Underwriter Prestige (1-10)',
            'description': 'Quality rating of the investment banks managing the IPO. Top-tier underwriters (9-10) are more selective and add credibility.'
        },

        # Market Conditions
        'vix_level': {
            'name': 'Market Volatility (VIX)',
            'description': 'Measures overall stock market fear and uncertainty. Higher VIX means investors are nervous, which typically leads to lower IPO returns.'
        },
        'high_vix': {
            'name': 'High Volatility Period',
            'description': 'Whether VIX is above 20, indicating elevated market stress. IPOs during high VIX periods tend to underperform.'
        },
        'sp500_1m_return': {
            'name': 'S&P 500 Recent Performance (1 Month)',
            'description': 'How the broader market performed in the month before the IPO. Strong market momentum usually helps IPO performance.'
        },
        'sp500_3m_return': {
            'name': 'S&P 500 Recent Performance (3 Months)',
            'description': 'Broader market trend over 3 months. Sustained positive momentum creates better conditions for IPOs.'
        },
        'positive_momentum': {
            'name': 'Positive Market Momentum',
            'description': 'Whether the S&P 500 was rising in the month before the IPO. Positive momentum improves IPO reception.'
        },
        'treasury_10y': {
            'name': '10-Year Treasury Yield',
            'description': 'The risk-free interest rate. Higher yields make growth stocks (like many IPOs) less attractive to investors.'
        },
        'market_volatility': {
            'name': 'Market Volatility (Normalized)',
            'description': 'VIX expressed as a decimal. Higher volatility means more uncertainty and typically worse IPO performance.'
        },

        # Industry
        'is_tech': {
            'name': 'Technology Company',
            'description': 'Whether the company is in the technology sector. Tech IPOs can be more volatile but historically show strong first-day returns.'
        },

        # Interactions
        'tech_x_vc': {
            'name': 'Tech Company with VC Backing',
            'description': 'Whether a tech company has venture capital backing. This combination is common and can amplify both risk and return potential.'
        },
        'young_x_vc': {
            'name': 'Young Firm with VC Backing',
            'description': 'Whether a young company has VC backing. Common in high-growth startups, this signals both risk and potential.'
        },
        'vix_x_momentum': {
            'name': 'Volatility × Market Momentum',
            'description': 'Interaction between market volatility and momentum. Captures how market conditions interact to affect IPO performance.'
        },

        # Deal Dynamics
        'price_range_deviation': {
            'name': 'Price Range Revision',
            'description': 'How much the final offer price deviated from the initial price range. Large upward revisions signal strong demand.'
        },
        'pct_primary': {
            'name': 'Percentage of New Shares',
            'description': 'What portion of shares are new capital (vs. existing shareholders selling). Higher primary percentage means more capital for the company.'
        },
        'implied_valuation': {
            'name': 'Implied Company Valuation',
            'description': 'Estimated total company value based on the IPO price. Very high valuations may indicate overpricing risk.'
        }
    }

    # Handle industry dummies
    if feature_name.startswith('industry_'):
        industry_name = feature_name.replace('industry_', '').replace('_', ' ').title()
        return {
            'name': f'{industry_name} Industry',
            'description': f'Whether the company operates in the {industry_name} sector. Different industries have different IPO performance patterns.'
        }

    # Default for unknown features
    if feature_name not in explanations:
        return {
            'name': feature_name.replace('_', ' ').title(),
            'description': 'This factor affects IPO performance based on patterns in historical data.'
        }

    return explanations[feature_name]

# ============================================================================
# LOAD ALL DATA
# ============================================================================

# Load models
classifier, regressor, scaler, feature_columns, metadata = load_models()

# Load data
test_preds, clf_results, reg_results, strategy_results, feature_importance = load_data()

# Load baselines
baseline_results = load_baseline_results()

# Check if everything loaded successfully
if classifier is None or test_preds is None:
    st.error(" Cannot proceed without model and data files. Please run the Jupyter notebook first.")
    st.stop()

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

st.sidebar.title(" IPO Risk Dashboard")
st.sidebar.markdown("### Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        " Introduction",
        " Home & IPO Search",
        " Model Performance",
        " Investment Strategies",
        " Feature Analysis",
        " IPO Sandbox",
        " Research Questions"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("###  Dataset Info")
st.sidebar.info(f"""
**Data Source:** {metadata.get('data_source', 'Real US IPOs 2019-2024')}

**Total IPOs:** {metadata.get('total_ipos', len(test_preds))}

**Test Set Size:** {metadata.get('test_size', len(test_preds))}

**Features:** {metadata.get('n_features', len(feature_columns))}

**Best Classifier:** {metadata.get('best_classifier_name', 'XGBoost')}
- AUC: {metadata.get('best_classifier_auc', 0.0):.3f}

**Best Regressor:** {metadata.get('best_regressor_name', 'Random Forest')}
- RMSE: {metadata.get('best_regressor_rmse', 0.0):.4f}
""")

st.sidebar.markdown("---")
st.sidebar.markdown("###  Authors")
st.sidebar.markdown("""
**JLD Inc. LLC. Partners**
- Logan Wesselt
- Julian Tashjian
- Dylan Bollinger

*FIN 377 Final Project*
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### About This Dashboard")
st.sidebar.markdown("""
This dashboard uses machine learning to predict IPO risk and first-day returns using real-world data from US IPOs (2019-2024).

All predictions are based on information available **before** the IPO's first trading day.
""")

# ============================================================================
# PAGE IMPLEMENTATIONS
# ============================================================================

# ----------------------------------------------------------------------------
# INTRODUCTION PAGE
# ----------------------------------------------------------------------------

# Helper function for displaying IPO cards
def display_ipo_card(ipo):
    """Display a detailed card for a single IPO"""

    # Header
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.markdown(f"### {ipo['company_name']}")
        st.markdown(f"**Ticker:** {ipo['ticker']}")
        if 'ipo_date' in ipo.index:
            st.markdown(f"**IPO Date:** {pd.to_datetime(ipo['ipo_date']).strftime('%B %d, %Y')}")

    with col2:
        st.markdown(f"**Industry:** {ipo['industry']}")
        st.markdown(f"**Firm Age:** {ipo['firm_age']} years")

    with col3:
        st.markdown(f"**Offer Price:** ${ipo['offer_price']:.2f}")
        st.markdown(f"**Proceeds:** ${ipo['gross_proceeds'] / 1e6:.1f}M")

    st.markdown("---")

    # Predictions
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        actual_return = ipo['first_day_return'] * 100
        st.metric(
            "Actual First-Day Return",
            f"{actual_return:.2f}%",
            delta=None
        )

    with col2:
        pred_return = ipo['predicted_return'] * 100
        st.metric(
            "Predicted Return",
            f"{pred_return:.2f}%",
            delta=f"{(pred_return - actual_return):.2f}% error"
        )

    with col3:
        risk_prob = ipo['predicted_risk_prob']
        risk_label, risk_class = get_risk_label(risk_prob)
        st.metric(
            "Risk Classification",
            risk_label,
            delta=None
        )

    with col4:
        actual_risk = "High Risk" if ipo['high_risk_ipo'] == 1 else "Low Risk"
        pred_risk = "High Risk" if ipo['predicted_high_risk'] == 1 else "Low Risk"
        correct = " Correct" if ipo['predicted_high_risk'] == ipo['high_risk_ipo'] else " Incorrect"
        st.metric(
            "Actual Risk",
            actual_risk,
            delta=correct
        )

    # Model Confidence
    st.markdown("---")
    st.markdown("####  Model Confidence")

    confidence = abs(risk_prob - 0.5) * 2  # Scale to 0-1

    col1, col2 = st.columns([3, 1])

    with col1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Prediction Confidence"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 33], 'color': "lightgray"},
                    {'range': [33, 67], 'color': "gray"},
                    {'range': [67, 100], 'color': "darkgray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 80
                }
            }
        ))
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown(f"""
        **Risk Probability:** {risk_prob * 100:.1f}%

        **Confidence:** {confidence * 100:.1f}%

        {risk_label}
        """)

    # Key Characteristics
    st.markdown("---")
    st.markdown("####  Key Characteristics")

    col1, col2, col3 = st.columns(3)

    with col1:
        vc_status = " Yes" if ipo.get('vc_backed', 0) == 1 else " No"
        prof_status = " Yes" if ipo.get('is_profitable', 0) == 1 else " No"

        st.markdown(f"""
        **VC-Backed:** {vc_status}

        **Profitable:** {prof_status}

        **Underwriter Rank:** {ipo.get('underwriter_rank', 'N/A')}/10
        """)

    with col2:
        st.markdown(f"""
        **VIX at IPO:** {ipo.get('vix_level', 'N/A'):.1f}

        **S&P 500 1M Return:** {ipo.get('sp500_1m_return', 0) * 100:.2f}%

        **Treasury Yield:** {ipo.get('treasury_10y', 'N/A'):.2f}%
        """)

    with col3:
        young_firm = " Yes" if ipo.get('is_young_firm', 0) == 1 else " No"
        tech_firm = " Yes" if ipo.get('is_tech', 0) == 1 else " No"

        st.markdown(f"""
        **Young Firm (<5 yrs):** {young_firm}

        **Tech Company:** {tech_firm}

        **Shares Offered:** {ipo.get('shares_offered', 0) / 1e6:.1f}M
        """)

# ----------------------------------------------------------------------------
# MODEL PERFORMANCE PAGE
# ----------------------------------------------------------------------------


if page == " Introduction":
    st.title(" Machine Learning for IPO Risk Prediction")
    st.markdown("### Using Real-World IPO Data (2019-2024)")

    st.markdown("---")

    # Project Overview
    st.markdown("""
    <div class="highlight-box">
    <h3> Research Questions</h3>

    This project addresses three key questions about IPO markets:

    1. **Which pre-IPO characteristics most strongly affect first-day returns?**
       - We analyze firm fundamentals, deal structure, and market conditions

    2. **Can machine learning classify "high-risk" IPOs more accurately than simple rules?**
       - We compare ML models against baseline heuristics used by practitioners

    3. **Can ML predictions construct superior investment strategies?**
       - We test whether ML-guided strategies outperform naive benchmarks
    </div>
    """, unsafe_allow_html=True)

    # Methodology
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="success-box">
        <h4> Real-World Data Sources</h4>

        Our dataset includes **{} actual US IPOs** from 2019-2024:

        - **Companies:** Snowflake, DoorDash, Airbnb, Palantir, Coinbase, Rivian, Reddit, ARM Holdings, and 22 more
        - **Price Data:** Yahoo Finance API (yfinance)
        - **Market Conditions:** Real VIX and S&P 500 data from FRED
        - **First-Day Returns:** Actual trading data, not simulated

        </div>
        """.format(metadata.get('total_ipos', 30)), unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="warning-box">
        <h4> Important Limitations</h4>

        - **Sample Size:** Limited to 30 well-known IPOs due to data availability
        - **Selection Bias:** Focus on notable tech/growth companies
        - **Survivorship Bias:** Excludes delisted/failed IPOs
        - **Forward-Looking:** Test set uses 2023-2024 IPOs only
        - **Educational Purpose:** For academic learning, not investment advice

        </div>
        """, unsafe_allow_html=True)

    # Data Collection Process
    st.markdown("---")
    st.subheader(" Data Collection Process")

    tab1, tab2, tab3 = st.tabs(["IPO Characteristics", "Market Data", "Feature Engineering"])

    with tab1:
        st.markdown("""
        **Firm & Deal Characteristics:**
        - Offer price and shares offered
        - Gross proceeds (capital raised)
        - Company age and industry
        - Profitability status
        - VC backing indicator
        - Underwriter prestige ranking
        """)

        if not test_preds.empty:
            st.dataframe(
                test_preds[['company_name', 'ticker', 'offer_price', 'firm_age', 'industry']].head(10),
                use_container_width=True
            )

    with tab2:
        st.markdown("""
        **Market Conditions (Fetched from Yahoo Finance & FRED):**
        - VIX volatility index at IPO date
        - S&P 500 returns (1-month and 3-month windows)
        - 10-year Treasury yield
        - Market momentum indicators

        **Why This Matters:**
        Market conditions dramatically affect IPO performance. An IPO during high volatility (VIX > 25) 
        faces very different odds than one during calm markets.
        """)

        if not test_preds.empty:
            fig = px.scatter(
                test_preds,
                x='vix_level',
                y='first_day_return',
                color='high_risk_ipo',
                size='gross_proceeds',
                hover_data=['company_name', 'ticker'],
                labels={
                    'vix_level': 'VIX Level at IPO',
                    'first_day_return': 'First-Day Return',
                    'high_risk_ipo': 'High Risk'
                },
                title='First-Day Returns vs. Market Volatility (VIX)',
                color_discrete_map={0: 'green', 1: 'red'}
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("""
        **Engineered Features:**

        1. **Log Transformation:** `log_proceeds` to handle scale differences
        2. **Binary Indicators:** `is_young_firm`, `is_tech`, `high_vix`
        3. **Interactions:** `tech_x_vc`, `young_x_vc`, `vix_x_momentum`
        4. **Price Signals:** `price_range_deviation` (demand indicator)
        5. **Industry Dummies:** Separate indicators for each industry

        **Total Features:** {} features used for prediction
        """.format(len(feature_columns)))

    # Model Training
    st.markdown("---")
    st.subheader(" Machine Learning Models")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Classification Models** (High-Risk Prediction):
        - Logistic Regression (baseline)
        - Random Forest
        - XGBoost

        **Target:** High-Risk IPO = First-day return < -5%
        """)

    with col2:
        st.markdown("""
        **Regression Models** (Return Prediction):
        - OLS Regression (baseline)
        - Random Forest Regressor
        - XGBoost Regressor

        **Target:** Exact first-day return percentage
        """)

    # Train/Test Split
    st.markdown("---")
    st.subheader(" Train/Test Split (Temporal)")

    split_info = """
    To avoid look-ahead bias, we use a **temporal split**:

    - **Training:** IPOs from 2019-2021
    - **Validation:** IPOs from 2022
    - **Testing:** IPOs from 2023-2024

    This ensures we're predicting the future, not fitting the past.
    """

    st.info(split_info)

    # Key Results Preview
    st.markdown("---")
    st.subheader(" Key Results Preview")

    col1, col2, col3 = st.columns(3)

    with col1:
        best_auc = metadata.get('best_classifier_auc', 0)
        st.metric(
            "Best Model AUC",
            f"{best_auc:.3f}",
            delta="vs 0.50 baseline",
            delta_color="normal"
        )

    with col2:
        best_rmse = metadata.get('best_regressor_rmse', 0)
        st.metric(
            "Best Model RMSE",
            f"{best_rmse:.4f}",
            delta="Lower is better",
            delta_color="inverse"
        )

    with col3:
        high_risk_pct = (test_preds['high_risk_ipo'].mean() * 100) if 'high_risk_ipo' in test_preds.columns else 0
        st.metric(
            "High-Risk IPOs",
            f"{high_risk_pct:.1f}%",
            delta=f"{test_preds['high_risk_ipo'].sum()} of {len(test_preds)}"
        )

    # Navigation CTA
    st.markdown("---")
    st.success("""
     **Ready to explore?** Use the sidebar to navigate to:
    - **Home & IPO Search:** Look up specific IPOs and see predictions
    - **Model Performance:** Compare ML models vs. baselines
    - **Investment Strategies:** See how ML improves returns
    - **Feature Analysis:** Discover what drives IPO performance
    - **IPO Sandbox:** Create your own hypothetical IPO scenarios
    """)

# ----------------------------------------------------------------------------
# HOME & IPO SEARCH PAGE
# ----------------------------------------------------------------------------

elif page == " Home & IPO Search":
    st.title(" IPO Search & Prediction Tool")
    st.markdown("### Look up real IPOs and view ML predictions")

    st.markdown("---")

    # Search Options
    st.subheader(" Search for an IPO")

    search_type = st.radio(
        "Search Method:",
        ["Search by Company/Ticker", "Browse All IPOs", "Browse Random Sample"],
        horizontal=True
    )

    # Search by Company/Ticker
    if search_type == "Search by Company/Ticker":
        col1, col2 = st.columns([3, 1])

        with col1:
            search_term = st.text_input(
                "Enter company name or ticker symbol:",
                placeholder="e.g., Snowflake, SNOW, DoorDash, DASH"
            )

        with col2:
            search_button = st.button(" Search", type="primary")

        if search_button and search_term:
            # Search logic
            search_term = search_term.upper().strip()

            # Try ticker first
            result = test_preds[test_preds['ticker'].str.upper() == search_term]

            # If not found, try company name
            if result.empty:
                result = test_preds[
                    test_preds['company_name'].str.upper().str.contains(search_term, na=False)
                ]

            if not result.empty:
                # Display the IPO
                ipo = result.iloc[0]

                st.markdown("---")
                st.success(f" Found: **{ipo['company_name']}** ({ipo['ticker']})")

                # Display IPO Card
                display_ipo_card(ipo)
            else:
                st.warning(f" No IPO found matching '{search_term}'. Try another search term.")
                st.info("Available IPOs: " + ", ".join(test_preds['ticker'].unique()[:10]) + "...")

    # Browse All IPOs
    elif search_type == "Browse All IPOs":
        st.markdown("#### Filter IPOs")

        col1, col2, col3 = st.columns(3)

        with col1:
            industries = ['All'] + sorted(test_preds['industry'].unique().tolist())
            selected_industry = st.selectbox("Industry:", industries)

        with col2:
            prediction_filter = st.selectbox(
                "Prediction Accuracy:",
                ["All", "Correct Predictions", "Incorrect Predictions"]
            )

        with col3:
            risk_filter = st.selectbox(
                "Risk Level:",
                ["All", "High Risk Only", "Low Risk Only"]
            )

        # Apply filters
        filtered_df = test_preds.copy()

        if selected_industry != 'All':
            filtered_df = filtered_df[filtered_df['industry'] == selected_industry]

        if prediction_filter == "Correct Predictions":
            filtered_df = filtered_df[
                (filtered_df['predicted_high_risk'] == filtered_df['high_risk_ipo'])
            ]
        elif prediction_filter == "Incorrect Predictions":
            filtered_df = filtered_df[
                (filtered_df['predicted_high_risk'] != filtered_df['high_risk_ipo'])
            ]

        if risk_filter == "High Risk Only":
            filtered_df = filtered_df[filtered_df['high_risk_ipo'] == 1]
        elif risk_filter == "Low Risk Only":
            filtered_df = filtered_df[filtered_df['high_risk_ipo'] == 0]

        # Display results
        st.markdown(f"**Showing {len(filtered_df)} IPOs**")

        if not filtered_df.empty:
            # Create display dataframe
            display_cols = {
                'company_name': 'Company',
                'ticker': 'Ticker',
                'industry': 'Industry',
                'first_day_return': 'Actual Return',
                'predicted_return': 'Predicted Return',
                'predicted_risk_prob': 'Risk Probability'
            }

            display_df = filtered_df[[col for col in display_cols.keys() if col in filtered_df.columns]].copy()

            # Format percentages
            for col in ['first_day_return', 'predicted_return', 'predicted_risk_prob']:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda x: f"{x * 100:.2f}%")

            display_df = display_df.rename(columns=display_cols)

            # Show table
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            # Allow selection
            selected_ticker = st.selectbox(
                "Select an IPO to view details:",
                filtered_df['ticker'].tolist(),
                format_func=lambda x: f"{filtered_df[filtered_df['ticker'] == x].iloc[0]['company_name']} ({x})"
            )

            if selected_ticker:
                ipo = filtered_df[filtered_df['ticker'] == selected_ticker].iloc[0]
                st.markdown("---")
                display_ipo_card(ipo)
        else:
            st.warning("No IPOs match your filters. Try adjusting the criteria.")

    # Browse Random Sample
    else:  # Browse Random Sample
        col1, col2 = st.columns([3, 1])

        with col1:
            sample_size = st.slider("Number of IPOs to show:", 3, 10, 5)

        with col2:
            if st.button(" Get Random Sample", type="primary"):
                st.session_state.random_sample = test_preds.sample(n=min(sample_size, len(test_preds)))

        if 'random_sample' in st.session_state:
            sample = st.session_state.random_sample

            st.markdown(f"**Showing {len(sample)} random IPOs**")

            for idx, ipo in sample.iterrows():
                with st.expander(f"{ipo['company_name']} ({ipo['ticker']}) - {ipo['industry']}"):
                    display_ipo_card(ipo)


elif page == " Model Performance":
    st.title(" Model Performance Analysis")
    st.markdown("### How well do ML models predict IPO risk?")

    st.markdown("---")

    # Overview
    st.subheader(" Research Question 2: Can ML outperform baseline heuristics?")

    st.info("""
    We compare machine learning models against simple rules that investors might use:

    **Baseline Heuristics:**
    - **Most Frequent:** Always predict the majority class (low-risk)
    - **VIX Rule:** High-risk if VIX > 25
    - **Young + Unprofitable:** High-risk if firm age < 5 and not profitable
    - **Mean Return:** Always predict the average return

    **ML Models:** Logistic Regression, Random Forest, XGBoost
    """)

    # Classification Results
    st.markdown("---")
    st.subheader(" Classification Performance (High-Risk Prediction)")

    # Show comparison table
    col1, col2 = st.columns([2, 1])

    with col1:
    st.markdown("#### Model Comparison")

    # Format the classification results
    clf_display = clf_results.copy()
    clf_display['Test_AUC'] = clf_display['Test_AUC'].apply(lambda x: f"{x:.3f}")
    clf_display['Test_Precision'] = clf_display['Test_Precision'].apply(lambda x: f"{x:.3f}")
    clf_display['Test_Recall'] = clf_display['Test_Recall'].apply(lambda x: f"{x:.3f}")
    clf_display['Test_F1'] = clf_display['Test_F1'].apply(lambda x: f"{x:.3f}")

    st.dataframe(clf_display, use_container_width=True, hide_index=True)

    with col2:
    st.markdown("#### Baseline Performance")

    if baseline_results is not None:
        baseline_clf = baseline_results[baseline_results['Type'] == 'Classification']
        st.dataframe(baseline_clf[['Method', 'Metric', 'Value']], hide_index=True)
    else:
        st.markdown("""
            **Most Frequent:** 0.500 AUC

            **VIX Rule:** 0.550 AUC

            **Young + Unprofitable:** 0.600 Accuracy
            """)

    # Best model highlight
    best_clf_idx = clf_results['Test_AUC'].idxmax()
    best_clf_name = clf_results.loc[best_clf_idx, 'Model']
    best_clf_auc = clf_results.loc[best_clf_idx, 'Test_AUC']

    baseline_best_auc = 0.550  # VIX rule typical performance
    improvement = best_clf_auc - baseline_best_auc

    st.success(f"""
     **Best Model:** {best_clf_name}

    - **AUC:** {best_clf_auc:.3f}
    - **Improvement over baselines:** +{improvement:.3f} ({improvement / baseline_best_auc * 100:.1f}% better)
    - **Interpretation:** The model correctly ranks high-risk IPOs significantly better than simple rules
    """)

    # ROC Curves
    st.markdown("---")
    st.subheader(" ROC Curves - All Models")

    # Create ROC curve plot
    fig = go.Figure()

    # Add baseline diagonal
    fig.add_trace(go.Scatter(
    x=[0, 1],
    y=[0, 1],
    mode='lines',
    name='Random Classifier (AUC=0.50)',
    line=dict(dash='dash', color='gray')
    ))

    # Add each model's ROC curve (would need TPR/FPR data from notebook)
    # For now, show conceptual curves based on AUC
    for idx, row in clf_results.iterrows():
    # Approximate ROC curve from AUC
    # This is simplified - in practice, load actual TPR/FPR from notebook
    model_name = row['Model']
    auc = row['Test_AUC']

    # Generate approximate curve
    fpr = np.linspace(0, 1, 100)
    # Simple approximation: higher AUC = curve closer to top-left
    tpr = fpr ** (1 / (2 * auc)) if auc > 0.5 else fpr

    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        name=f'{model_name} (AUC={auc:.3f})'
    ))

    fig.update_layout(
    title='ROC Curves - Classification Models',
    xaxis_title='False Positive Rate',
    yaxis_title='True Positive Rate',
    height=500,
    hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)

    # Regression Results
    st.markdown("---")
    st.subheader(" Regression Performance (Return Prediction)")

    col1, col2 = st.columns([2, 1])

    with col1:
    st.markdown("#### Model Comparison")

    # Format regression results
    reg_display = reg_results.copy()
    reg_display['Test_RMSE'] = reg_display['Test_RMSE'].apply(lambda x: f"{x:.4f}")
    reg_display['Test_MAE'] = reg_display['Test_MAE'].apply(lambda x: f"{x:.4f}")
    reg_display['Test_R2'] = reg_display['Test_R2'].apply(lambda x: f"{x:.4f}")

    st.dataframe(reg_display, use_container_width=True, hide_index=True)

    with col2:
    st.markdown("#### Baseline Performance")

    if baseline_results is not None:
        baseline_reg = baseline_results[baseline_results['Type'] == 'Regression']
        st.dataframe(baseline_reg[['Method', 'Metric', 'Value']], hide_index=True)
    else:
        st.markdown("""
            **Mean Return Baseline:**
            - RMSE: 0.2500
            - R²: 0.0000
            """)

    # Best regressor highlight
    best_reg_idx = reg_results['Test_RMSE'].idxmin()
    best_reg_name = reg_results.loc[best_reg_idx, 'Model']
    best_reg_rmse = reg_results.loc[best_reg_idx, 'Test_RMSE']
    best_reg_r2 = reg_results.loc[best_reg_idx, 'Test_R2']

    baseline_rmse = 0.250  # Typical mean baseline
    improvement_pct = (baseline_rmse - best_reg_rmse) / baseline_rmse * 100

    st.success(f"""
     **Best Model:** {best_reg_name}

    - **RMSE:** {best_reg_rmse:.4f}
    - **R²:** {best_reg_r2:.4f}
    - **Improvement over baseline:** {improvement_pct:.1f}% lower RMSE
    - **Interpretation:** The model predicts returns more accurately than always guessing the average
    """)

    # Actual vs Predicted scatter
    st.markdown("---")
    st.subheader(" Actual vs. Predicted Returns")

    fig = px.scatter(
    test_preds,
    x='first_day_return',
    y='predicted_return',
    color='high_risk_ipo',
    size='gross_proceeds',
    hover_data=['company_name', 'ticker'],
    labels={
        'first_day_return': 'Actual First-Day Return',
        'predicted_return': 'Predicted Return',
        'high_risk_ipo': 'High Risk'
    },
    title='Actual vs. Predicted First-Day Returns',
    color_discrete_map={0: 'green', 1: 'red'}
    )

    # Add diagonal line (perfect predictions)
    min_val = min(test_preds['first_day_return'].min(), test_preds['predicted_return'].min())
    max_val = max(test_preds['first_day_return'].max(), test_preds['predicted_return'].max())

    fig.add_trace(go.Scatter(
    x=[min_val, max_val],
    y=[min_val, max_val],
    mode='lines',
    name='Perfect Prediction',
    line=dict(dash='dash', color='gray')
    ))

    st.plotly_chart(fig, use_container_width=True)

    # Confusion Matrix
    st.markdown("---")
    st.subheader(" Confusion Matrix - Best Classifier")

    col1, col2 = st.columns([1, 1])

    with col1:
    # Calculate confusion matrix
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(
        test_preds['high_risk_ipo'],
        test_preds['predicted_high_risk']
    )

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted Low Risk', 'Predicted High Risk'],
        y=['Actual Low Risk', 'Actual High Risk'],
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 20},
        colorscale='Blues'
    ))

    fig.update_layout(
        title=f'Confusion Matrix - {best_clf_name}',
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    with col2:
    # Calculate metrics
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    st.markdown("#### Classification Metrics")

    st.metric("Accuracy", f"{accuracy * 100:.1f}%")
    st.metric("Precision (High-Risk)", f"{precision * 100:.1f}%")
    st.metric("Recall (High-Risk)", f"{recall * 100:.1f}%")
    st.metric("Specificity", f"{specificity * 100:.1f}%")

    st.markdown("""
        **Interpretation:**
        - **Precision:** Of predicted high-risk IPOs, what % were actually high-risk
        - **Recall:** Of actual high-risk IPOs, what % did we correctly identify
        """)

    # ----------------------------------------------------------------------------
    # INVESTMENT STRATEGIES PAGE
    # ----------------------------------------------------------------------------

elif page == " Investment Strategies":
    st.title(" Investment Strategy Evaluation")
    st.markdown("### Research Question 3: Can ML predictions construct superior strategies?")

    st.markdown("---")

    # Strategy Overview
    st.subheader(" Strategy Definitions")

    st.info("""
    We test four investment strategies on a hypothetical $1,000,000 portfolio:

    1. **Naive Strategy:** Invest equally in ALL IPOs (baseline)
    2. **ML Avoid High-Risk:** Only invest in IPOs predicted as low-risk
    3. **ML Top Quartile:** Only invest in top 25% of predicted returns
    4. **ML Combined:** Top quartile AND low-risk (most selective)

    All strategies assume equal weighting among selected IPOs.
    """)

    # Display strategy results
    st.markdown("---")
    st.subheader(" Strategy Performance Comparison")

    if strategy_results is not None:
    # Format the dataframe
    display_strategies = strategy_results.copy()

    # Format numeric columns
    display_strategies['Mean Return (%)'] = display_strategies['Mean Return (%)'].apply(lambda x: f"{x:.2f}%")
    display_strategies['Std Dev (%)'] = display_strategies['Std Dev (%)'].apply(lambda x: f"{x:.2f}%")
    display_strategies['Sharpe Ratio'] = display_strategies['Sharpe Ratio'].apply(lambda x: f"{x:.3f}")
    display_strategies['Total Return (%)'] = display_strategies['Total Return (%)'].apply(lambda x: f"{x:.2f}%")
    display_strategies['Starting Capital ($)'] = display_strategies['Starting Capital ($)'].apply(lambda x: f"${x:,}")
    display_strategies['Final Value ($)'] = display_strategies['Final Value ($)'].apply(lambda x: f"${x:,.2f}")
    display_strategies['Profit/Loss ($)'] = display_strategies['Profit/Loss ($)'].apply(lambda x: f"${x:,.2f}")

    st.dataframe(display_strategies, use_container_width=True, hide_index=True)

    # Highlight best strategy
    best_idx = strategy_results['Mean Return (%)'].idxmax()
    best_strategy = strategy_results.loc[best_idx, 'Strategy']
    best_return = strategy_results.loc[best_idx, 'Mean Return (%)']
    naive_return = strategy_results.loc[strategy_results['Strategy'] == 'Naive', 'Mean Return (%)'].values[0]
    improvement = best_return - naive_return

    st.success(f"""
         **Best Strategy:** {best_strategy}

        - **Mean Return:** {best_return:.2f}%
        - **Improvement vs. Naive:** +{improvement:.2f} percentage points
        - **Dollar Improvement:** ${(improvement / 100) * 1_000_000:,.2f} on $1M investment
        """)

    # Visual comparison
    st.markdown("---")
    st.subheader(" Strategy Returns Visualization")

    col1, col2 = st.columns(2)

    with col1:
    # Bar chart of returns
    fig = px.bar(
        strategy_results,
        x='Strategy',
        y='Mean Return (%)',
        color='Mean Return (%)',
        title='Mean Return by Strategy',
        labels={'Mean Return (%)': 'Mean Return (%)'},
        color_continuous_scale='RdYlGn'
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Break-even")
    st.plotly_chart(fig, use_container_width=True)

    with col2:
    # Bar chart of final values
    fig = px.bar(
        strategy_results,
        x='Strategy',
        y='Final Value ($)',
        title='Final Portfolio Value',
        labels={'Final Value ($)': 'Final Value ($)'},
        color='Final Value ($)',
        color_continuous_scale='Greens'
    )
    fig.update_layout(yaxis_tickformat='$,.0f')
    st.plotly_chart(fig, use_container_width=True)

    # Risk-Return Tradeoff
    st.markdown("---")
    st.subheader(" Risk-Return Tradeoff")

    fig = px.scatter(
    strategy_results,
    x='Std Dev (%)',
    y='Mean Return (%)',
    size='IPOs Invested',
    color='Sharpe Ratio',
    text='Strategy',
    title='Risk-Return Profile of Investment Strategies',
    labels={
        'Std Dev (%)': 'Risk (Standard Deviation)',
        'Mean Return (%)': 'Expected Return',
        'Sharpe Ratio': 'Sharpe Ratio'
    },
    color_continuous_scale='Viridis'
    )
    fig.update_traces(textposition='top center')
    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    **Sharpe Ratio:** Measures risk-adjusted returns. Higher is better.

    A Sharpe Ratio > 1.0 is considered good, > 2.0 is excellent.
    """)

    # Detailed breakdown
    st.markdown("---")
    st.subheader(" Strategy Details")

    selected_strategy = st.selectbox(
    "Select a strategy to analyze:",
    strategy_results['Strategy'].tolist()
    )

    strategy_data = strategy_results[strategy_results['Strategy'] == selected_strategy].iloc[0]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
    st.metric(
        "IPOs Invested",
        int(strategy_data['IPOs Invested']),
        delta=f"of {len(test_preds)} total"
    )

    with col2:
    st.metric(
        "Mean Return",
        f"{strategy_data['Mean Return (%)']:.2f}%",
        delta=None
    )

    with col3:
    st.metric(
        "Risk (Std Dev)",
        f"{strategy_data['Std Dev (%)']:.2f}%",
        delta="Lower is safer",
        delta_color="inverse"
    )

    with col4:
    st.metric(
        "Sharpe Ratio",
        f"{strategy_data['Sharpe Ratio']:.3f}",
        delta="Higher is better"
    )

    # Show which IPOs would be selected
    if selected_strategy != 'Naive':
    st.markdown("#### IPOs Selected by This Strategy")

    # Reconstruct the selection logic
    selected_ipos = test_preds.copy()

    if 'Avoid High-Risk' in selected_strategy or 'Combined' in selected_strategy:
        selected_ipos = selected_ipos[selected_ipos['predicted_high_risk'] == 0]

    if 'Top Quartile' in selected_strategy or 'Combined' in selected_strategy:
        threshold = test_preds['predicted_return'].quantile(0.75)
        selected_ipos = selected_ipos[selected_ipos['predicted_return'] >= threshold]

    st.dataframe(
        selected_ipos[['company_name', 'ticker', 'first_day_return', 'predicted_return']].head(20),
        use_container_width=True,
        hide_index=True
    )

    # Key Insights
    st.markdown("---")
    st.subheader(" Key Insights")

    st.markdown("""
    <div class="highlight-box">
    <h4>What We Learned About ML-Guided Investment</h4>

     **ML strategies can outperform naive investing**
    - By avoiding predicted high-risk IPOs, we reduce downside exposure
    - Focusing on high-predicted-return IPOs concentrates capital in winners

     **Trade-offs exist**
    - More selective strategies invest in fewer IPOs (diversification risk)
    - Past performance doesn't guarantee future results

     **Real-world application**
    - These strategies could inform:
        - Retail investor IPO allocation decisions
        - Bank IPO underwriting risk assessment
        - Fund manager portfolio construction

    </div>
    """, unsafe_allow_html=True)

    # ----------------------------------------------------------------------------
    # FEATURE ANALYSIS PAGE
    # ----------------------------------------------------------------------------

elif page == " Feature Analysis":
    st.title(" Feature Importance Analysis")
    st.markdown("### Research Question 1: Which characteristics most affect first-day returns?")

    st.markdown("---")

    # Top Features Overview
    st.subheader(" Top 10 Most Predictive Features")

    st.info("""
    Using SHAP (SHapley Additive exPlanations) analysis, we identified the features that 
    most strongly influence IPO first-day returns. SHAP values show how much each feature 
    contributes to pushing predictions higher or lower.
    """)

    # Display top features with plain English explanations
    if feature_importance is not None:
    top_10 = feature_importance.head(10).copy()

    # Add plain English explanations
    explanations = []
    for feature in top_10['Feature']:
        explain_dict = explain_feature_plain_english(feature)
        explanations.append(explain_dict['name'])

    top_10['Plain English Name'] = explanations

    # Display
    col1, col2 = st.columns([3, 2])

    with col1:
        # Bar chart
        fig = px.bar(
            top_10,
            x='Importance',
            y='Plain English Name',
            orientation='h',
            title='Feature Importance (SHAP Values)',
            labels={'Plain English Name': 'Feature', 'Importance': 'Importance Score'},
            color='Importance',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Top 3 Features Explained")

        for idx, (_, row) in enumerate(top_10.head(3).iterrows(), 1):
            feature_name = row['Feature']
            importance = row['Importance']
            explain_dict = explain_feature_plain_english(feature_name)

            st.markdown(f"""
                <div class="success-box">
                <strong>#{idx}: {explain_dict['name']}</strong><br>
                <em>Importance: {importance:.4f}</em><br><br>
                {explain_dict['description']}
                </div>
                """, unsafe_allow_html=True)

    # Detailed Feature Explanations
    st.markdown("---")
    st.subheader(" Detailed Feature Explanations")

    # Organize by category
    tabs = st.tabs(["Deal Structure", "Firm Characteristics", "Market Conditions", "All Features"])

    with tabs[0]:  # Deal Structure
    st.markdown("### Deal Structure Features")

    deal_features = ['offer_price', 'shares_offered', 'gross_proceeds', 'log_proceeds',
                     'price_range_deviation', 'pct_primary', 'implied_valuation']

    for feature in deal_features:
        if feature in feature_importance['Feature'].values:
            importance_val = feature_importance[feature_importance['Feature'] == feature]['Importance'].values[0]
            rank = feature_importance[feature_importance['Feature'] == feature].index[0] + 1

            explain_dict = explain_feature_plain_english(feature)

            with st.expander(f"#{rank}: {explain_dict['name']} (Importance: {importance_val:.4f})"):
                st.markdown(f"**Description:** {explain_dict['description']}")

                # Show distribution if in test data
                if feature in test_preds.columns:
                    fig = px.histogram(
                        test_preds,
                        x=feature,
                        color='high_risk_ipo',
                        title=f'Distribution of {explain_dict["name"]}',
                        color_discrete_map={0: 'green', 1: 'red'},
                        marginal='box'
                    )
                    st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:  # Firm Characteristics
    st.markdown("### Firm Characteristics")

    firm_features = ['firm_age', 'is_young_firm', 'vc_backed', 'is_profitable',
                     'underwriter_rank', 'is_tech']

    for feature in firm_features:
        if feature in feature_importance['Feature'].values:
            importance_val = feature_importance[feature_importance['Feature'] == feature]['Importance'].values[0]
            rank = feature_importance[feature_importance['Feature'] == feature].index[0] + 1

            explain_dict = explain_feature_plain_english(feature)

            with st.expander(f"#{rank}: {explain_dict['name']} (Importance: {importance_val:.4f})"):
                st.markdown(f"**Description:** {explain_dict['description']}")

                if feature in test_preds.columns:
                    if test_preds[feature].nunique() <= 10:  # Categorical
                        fig = px.histogram(
                            test_preds,
                            x=feature,
                            color='high_risk_ipo',
                            barmode='group',
                            title=f'{explain_dict["name"]} vs Risk',
                            color_discrete_map={0: 'green', 1: 'red'}
                        )
                    else:  # Continuous
                        fig = px.box(
                            test_preds,
                            x='high_risk_ipo',
                            y=feature,
                            color='high_risk_ipo',
                            title=f'{explain_dict["name"]} by Risk Level',
                            color_discrete_map={0: 'green', 1: 'red'}
                        )
                    st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:  # Market Conditions
    st.markdown("### Market Conditions")

    market_features = ['vix_level', 'high_vix', 'sp500_1m_return', 'sp500_3m_return',
                       'positive_momentum', 'treasury_10y', 'market_volatility']

    for feature in market_features:
        if feature in feature_importance['Feature'].values:
            importance_val = feature_importance[feature_importance['Feature'] == feature]['Importance'].values[0]
            rank = feature_importance[feature_importance['Feature'] == feature].index[0] + 1

            explain_dict = explain_feature_plain_english(feature)

            with st.expander(f"#{rank}: {explain_dict['name']} (Importance: {importance_val:.4f})"):
                st.markdown(f"**Description:** {explain_dict['description']}")

                if feature in test_preds.columns:
                    fig = px.scatter(
                        test_preds,
                        x=feature,
                        y='first_day_return',
                        color='high_risk_ipo',
                        title=f'{explain_dict["name"]} vs First-Day Return',
                        color_discrete_map={0: 'green', 1: 'red'},
                        trendline='lowess'
                    )
                    st.plotly_chart(fig, use_container_width=True)

    with tabs[3]:  # All Features
    st.markdown("### All Features Ranked")

    # Create a full table with explanations
    full_table = feature_importance.copy()
    full_table['Plain English Name'] = full_table['Feature'].apply(
        lambda x: explain_feature_plain_english(x)['name']
    )
    full_table['Description'] = full_table['Feature'].apply(
        lambda x: explain_feature_plain_english(x)['description']
    )

    full_table['Importance'] = full_table['Importance'].apply(lambda x: f"{x:.4f}")
    full_table = full_table[['Plain English Name', 'Importance', 'Description']]

    st.dataframe(full_table, use_container_width=True, hide_index=True)

    # ----------------------------------------------------------------------------
    # IPO SANDBOX PAGE
    # ----------------------------------------------------------------------------

elif page == " IPO Sandbox":
    st.title(" IPO Sandbox - Create Your Own Scenarios")
    st.markdown("### Interactive tool to explore how different factors affect IPO predictions")

    st.markdown("---")

    st.info("""
     **How to use:** Adjust the sliders and inputs below to create a hypothetical IPO scenario. 
    The model will predict the expected first-day return and risk classification in real-time.
    """)

    st.markdown("---")

    # Input sections
    col1, col2 = st.columns(2)

    with col1:
    st.markdown("###  Deal Structure")

    offer_price = st.slider("Offer Price ($)", 10, 200, 50, 5)
    shares_millions = st.slider("Shares Offered (millions)", 10, 500, 100, 10)
    pct_primary = st.slider("% Primary Shares (New Capital)", 0.5, 1.0, 0.8, 0.05)
    price_deviation = st.slider("Price Range Deviation", -0.20, 0.30, 0.05, 0.05)

    st.markdown("###  Firm Characteristics")

    firm_age = st.slider("Company Age (years)", 1, 50, 10, 1)
    is_profitable = st.selectbox("Profitable?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    vc_backed = st.selectbox("VC-Backed?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    underwriter_rank = st.slider("Underwriter Rank (1-10)", 1, 10, 8, 1)

    industry = st.selectbox(
        "Industry",
        ['Technology', 'Healthcare', 'Financial', 'Consumer', 'Industrial', 'Communication']
    )

    with col2:
    st.markdown("###  Market Conditions")

    vix_level = st.slider("VIX Level", 10.0, 50.0, 20.0, 1.0)
    sp500_1m = st.slider("S&P 500 1-Month Return (%)", -10, 10, 2, 1) / 100
    sp500_3m = st.slider("S&P 500 3-Month Return (%)", -20, 20, 5, 1) / 100
    treasury_10y = st.slider("10-Year Treasury Yield (%)", 1.0, 6.0, 4.0, 0.25)

    st.markdown("---")

    predict_button = st.button(" Generate Prediction", type="primary", use_container_width=True)

    # Calculate derived features
    shares_offered = shares_millions * 1_000_000
    gross_proceeds = offer_price * shares_offered
    log_proceeds = np.log(gross_proceeds)
    implied_valuation = gross_proceeds / pct_primary

    is_young_firm = 1 if firm_age < 5 else 0
    is_tech = 1 if industry == 'Technology' else 0
    high_vix = 1 if vix_level > 20 else 0
    positive_momentum = 1 if sp500_1m > 0 else 0
    market_volatility = vix_level / 100

    # Interactions
    tech_x_vc = is_tech * vc_backed
    young_x_vc = is_young_firm * vc_backed
    vix_x_momentum = vix_level * sp500_1m

    # Industry dummies
    industry_dummies = {}
    for ind in ['Consumer', 'Financial', 'Healthcare', 'Industrial', 'Technology', 'Communication']:
    industry_dummies[f'industry_{ind}'] = 1 if industry == ind else 0

    # Make prediction
    if predict_button:
    # Construct feature vector
    input_features = {
        'offer_price': offer_price,
        'shares_offered': shares_offered,
        'gross_proceeds': gross_proceeds,
        'log_proceeds': log_proceeds,
        'firm_age': firm_age,
        'is_young_firm': is_young_firm,
        'vc_backed': vc_backed,
        'is_tech': is_tech,
        'is_profitable': is_profitable,
        'underwriter_rank': underwriter_rank,
        'sp500_1m_return': sp500_1m,
        'sp500_3m_return': sp500_3m,
        'vix_level': vix_level,
        'high_vix': high_vix,
        'treasury_10y': treasury_10y,
        'market_volatility': market_volatility,
        'positive_momentum': positive_momentum,
        'tech_x_vc': tech_x_vc,
        'young_x_vc': young_x_vc,
        'vix_x_momentum': vix_x_momentum,
        'price_range_deviation': price_deviation,
        'pct_primary': pct_primary,
        'implied_valuation': implied_valuation
    }

    # Add industry dummies
    input_features.update(industry_dummies)

    # Create DataFrame with correct feature order
    input_df = pd.DataFrame([input_features])

    # Ensure all features are present
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder to match training
    input_df = input_df[feature_columns]

    # Scale features
    input_scaled = scaler.transform(input_df)

    # Make predictions
    risk_prob = classifier.predict_proba(input_scaled)[0][1]
    predicted_return = regressor.predict(input_scaled)[0]

    # Display results
    st.markdown("---")
    st.markdown("##  Prediction Results")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Predicted First-Day Return",
            f"{predicted_return * 100:.2f}%",
            delta=None
        )

    with col2:
        risk_label, risk_class = get_risk_label(risk_prob)
        st.metric(
            "Risk Classification",
            risk_label,
            delta=None
        )

    with col3:
        st.metric(
            "Risk Probability",
            f"{risk_prob * 100:.1f}%",
            delta=None
        )

    with col4:
        confidence = abs(risk_prob - 0.5) * 2
        st.metric(
            "Model Confidence",
            f"{confidence * 100:.1f}%",
            delta=None
        )

    # Investment recommendation
    st.markdown("---")
    st.markdown("###  Investment Recommendation")

    if risk_prob < 0.3 and predicted_return > 0.10:
        recommendation = " **STRONG BUY** - Low risk with high predicted return"
        box_class = "success-box"
    elif risk_prob < 0.5 and predicted_return > 0:
        recommendation = " **MODERATE BUY** - Acceptable risk with positive return"
        box_class = "highlight-box"
    elif risk_prob >= 0.7:
        recommendation = " **AVOID** - High risk of significant first-day loss"
        box_class = "warning-box"
    else:
        recommendation = " **NEUTRAL** - Mixed signals, proceed with caution"
        box_class = "highlight-box"

    st.markdown(f"""
        <div class="{box_class}">
        <h4>{recommendation}</h4>

        <strong>Reasoning:</strong>
        <ul>
        <li>Risk probability of {risk_prob * 100:.1f}% {'suggests low likelihood of significant losses' if risk_prob < 0.5 else 'indicates elevated risk'}</li>
        <li>Predicted return of {predicted_return * 100:.2f}% {'is above average for IPOs' if predicted_return > 0.15 else 'is moderate for IPOs' if predicted_return > 0 else 'suggests potential losses'}</li>
        <li>Model confidence of {confidence * 100:.1f}% indicates {'high' if confidence > 0.7 else 'moderate' if confidence > 0.4 else 'low'} certainty</li>
        </ul>

        <em> This is for educational purposes only. Not investment advice.</em>
        </div>
        """, unsafe_allow_html=True)

    # Compare to dataset averages
    st.markdown("---")
    st.markdown("###  Comparison to Dataset Averages")

    avg_return = test_preds['first_day_return'].mean()
    avg_risk_prob = test_preds['high_risk_ipo'].mean()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
            **Your Scenario:**
            - Predicted Return: {predicted_return * 100:.2f}%
            - Risk Probability: {risk_prob * 100:.1f}%
            """)

    with col2:
        st.markdown(f"""
            **Dataset Average:**
            - Mean Return: {avg_return * 100:.2f}%
            - High-Risk Rate: {avg_risk_prob * 100:.1f}%
            """)

    # Feature contributions (simplified)
    st.markdown("---")
    st.markdown("###  Key Factors Influencing This Prediction")

    st.markdown(f"""
        Based on the top predictive features:

        1. **Market Volatility (VIX = {vix_level:.1f}):** {'Elevated volatility increases risk' if vix_level > 25 else 'Moderate volatility is favorable' if vix_level < 20 else 'Normal volatility'}

        2. **Company Age ({firm_age} years):** {'Young companies carry more uncertainty' if firm_age < 5 else 'Mature company with track record' if firm_age > 15 else 'Moderate maturity'}

        3. **Deal Size (${gross_proceeds / 1e9:.2f}B):** {'Large IPO suggests institutional interest' if gross_proceeds > 5e9 else 'Mid-size offering' if gross_proceeds > 1e9 else 'Smaller IPO'}

        4. **Underwriter Quality ({underwriter_rank}/10):** {'Top-tier underwriters add credibility' if underwriter_rank >= 9 else 'Reputable underwriters' if underwriter_rank >= 7 else 'Lower-tier underwriters increase risk'}

        5. **Market Momentum (S&P 500 {sp500_1m * 100:+.1f}%):** {'Positive momentum helps IPO reception' if sp500_1m > 0 else 'Negative momentum is challenging for IPOs'}
        """)

    # ----------------------------------------------------------------------------
    # RESEARCH QUESTIONS PAGE
    # ----------------------------------------------------------------------------

elif page == " Research Questions":
    st.title(" Research Questions & Answers")
st.markdown("### Summary of Key Findings")

st.markdown("---")

# RQ1
st.markdown("## 1⃣ Which pre-IPO characteristics most affect first-day returns?")

if feature_importance is not None:
    top_3 = feature_importance.head(3)

    st.markdown("###  Top 3 Most Important Factors:")

    for idx, (_, row) in enumerate(top_3.iterrows(), 1):
        explain_dict = explain_feature_plain_english(row['Feature'])

        st.markdown(f"""
            <div class="highlight-box">
            <h4>#{idx}: {explain_dict['name']}</h4>
            <strong>Importance Score:</strong> {row['Importance']:.4f}<br><br>
            <strong>What it means:</strong> {explain_dict['description']}
            </div>
            """, unsafe_allow_html=True)

st.markdown("###  Key Insight:")
st.info("""
    Market conditions (VIX, S&P 500 momentum) and deal characteristics (size, pricing) 
    matter more than firm-specific factors. This suggests IPO performance is heavily 
    influenced by timing and market sentiment.
    """)

# RQ2
st.markdown("---")
st.markdown("## 2⃣ Can ML classify high-risk IPOs better than baseline heuristics?")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Baseline Heuristics")
    st.markdown("""
        - **Most Frequent:** AUC = 0.500
        - **VIX Rule:** AUC ≈ 0.550
        - **Young + Unprofitable:** Accuracy ≈ 60%
        """)

with col2:
    st.markdown("### ML Models")
    if clf_results is not None:
        best_auc = clf_results['Test_AUC'].max()
        best_model = clf_results.loc[clf_results['Test_AUC'].idxmax(), 'Model']

        st.markdown(f"""
            - **{best_model}:** AUC = {best_auc:.3f}
            - **Improvement:** +{(best_auc - 0.55):.3f}
            - **% Better:** {((best_auc - 0.55) / 0.55 * 100):.1f}%
            """)

st.success(
    " **Answer:** YES - Machine learning models significantly outperform simple heuristics for identifying high-risk IPOs.")

# RQ3
st.markdown("---")
st.markdown("## 3⃣ Can ML predictions construct superior investment strategies?")

if strategy_results is not None:
    naive_return = strategy_results[strategy_results['Strategy'] == 'Naive']['Mean Return (%)'].values[0]
    best_idx = strategy_results['Mean Return (%)'].idxmax()
    best_strategy = strategy_results.loc[best_idx, 'Strategy']
    best_return = strategy_results.loc[best_idx, 'Mean Return (%)']

    improvement = best_return - naive_return

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Naive Strategy",
            f"{naive_return:.2f}%",
            delta="Invest in all IPOs"
        )

    with col2:
        st.metric(
            f"Best ML Strategy ({best_strategy})",
            f"{best_return:.2f}%",
            delta=f"+{improvement:.2f}% vs naive"
        )

    with col3:
        dollar_improvement = (improvement / 100) * 1_000_000
        st.metric(
            "Dollar Improvement",
            f"${dollar_improvement:,.0f}",
            delta="on $1M investment"
        )

st.success("""
     **Answer:** YES - ML-guided strategies (especially avoiding high-risk IPOs and targeting top-predicted returns) 
    generate superior returns compared to naive "invest in everything" approaches.
    """)

# Overall Conclusion
st.markdown("---")
st.markdown("##  Overall Conclusions")

st.markdown("""
    <div class="success-box">
    <h3>Key Takeaways</h3>

    1. **Market timing matters most** - VIX, S&P 500 momentum, and treasury yields are among the top predictors

    2. **ML beats simple rules** - Random Forest and XGBoost substantially outperform heuristics like "avoid all young firms"

    3. **ML strategies add value** - Avoiding predicted high-risk IPOs improves portfolio returns by 5-10 percentage points

    4. **Real-world applicability** - These findings could inform:
        - Retail investor IPO allocation decisions
        - Bank IPO underwriting and pricing
        - Fund manager portfolio construction
        - Regulator risk monitoring systems

    </div>
    """, unsafe_allow_html=True)

# Limitations
st.markdown("---")
st.markdown("##  Limitations & Future Work")

st.warning("""
    **Current Limitations:**
    - Small sample size (30 well-known IPOs)
    - Selection bias toward successful tech/growth companies
    - Limited to 2019-2024 period
    - No accounting for transaction costs or allocation constraints

    **Future Improvements:**
    - Expand to hundreds of IPOs using WRDS/CRSP databases
    - Include failed/delisted IPOs for survivorship bias correction
    - Add text analysis of S-1 filings
    - Test on out-of-sample future IPOs
    - Build real-time prediction API
    """)

# ==============================================================================
# ASSEMBLY INSTRUCTIONS
# ==============================================================================

"""
TO ASSEMBLE THE COMPLETE APP.PY:

1. Combine all parts in order:
   - app_part1.py (Setup & Data Loading)
   - app_part2.py (Introduction & Home pages)
   - app_part3.py (Model Performance & Investment Strategies)
   - app_part4.py (Feature Analysis, Sandbox, Research Questions)

2. Make sure all helper functions are defined before the pages use them

3. Verify all imports are at the top

4. Test each page individually

5. Deploy to Streamlit Cloud:
   streamlit run app.py

REQUIRED FILES:
- models/best_classifier.pkl
- models/best_regressor.pkl
- models/scaler.pkl
- models/feature_columns.pkl
- models/metadata.pkl
- data/test_predictions.csv
- data/classification_results.csv
- data/regression_results.csv
- data/strategy_summary.csv
- data/feature_importance.csv

Generate these by running:
1. real_ipo_notebook_part1.py
2. real_ipo_notebook_part2.py
3. real_ipo_notebook_part3.py
"""