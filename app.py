"""
IPO Risk Prediction Dashboard - Part 1
Setup, Imports, and Data Loading

Authors: Logan Wesselt, Julian Tashjian, Dylan Bollinger
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="IPO Risk Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {padding: 0rem 1rem;}
    .stMetric {background-color: #f0f2f6; padding: 15px; border-radius: 10px;}
    h1 {color: #1f77b4;}
    .highlight-box {background-color: #e8f4f8; padding: 20px; border-radius: 10px; border-left: 5px solid #1f77b4;}
    .warning-box {background-color: #fff3cd; padding: 15px; border-radius: 8px; border-left: 5px solid #ffc107;}
    .success-box {background-color: #d4edda; padding: 15px; border-radius: 8px; border-left: 5px solid #28a745;}
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_models():
    try:
        with open('models/best_classifier.pkl', 'rb') as f:
            classifier = pickle.load(f)
        with open('models/best_regressor.pkl', 'rb') as f:
            regressor = pickle.load(f)
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('models/feature_columns.pkl', 'rb') as f:
            features = pickle.load(f)
        with open('models/metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        return classifier, regressor, scaler, features, metadata
    except FileNotFoundError:
        st.error("Model files not found. Run notebook first.")
        return None, None, None, None, None

@st.cache_data
def load_data():
    try:
        test_preds = pd.read_csv('data/test_predictions.csv')
        clf_results = pd.read_csv('data/classification_results.csv')
        reg_results = pd.read_csv('data/regression_results.csv')
        strategy_results = pd.read_csv('data/strategy_summary.csv')
        feature_importance = pd.read_csv('data/feature_importance.csv')
        return test_preds, clf_results, reg_results, strategy_results, feature_importance
    except FileNotFoundError:
        st.error("Data files not found. Run notebook first.")
        return None, None, None, None, None

def get_risk_label(risk_prob):
    if risk_prob >= 0.7:
        return "High Risk"
    elif risk_prob >= 0.4:
        return "Moderate Risk"
    else:
        return "Low Risk"

classifier, regressor, scaler, feature_columns, metadata = load_models()
test_preds, clf_results, reg_results, strategy_results, feature_importance = load_data()

if classifier is None or test_preds is None:
    st.stop()

st.sidebar.title("IPO Risk Dashboard")
page = st.sidebar.radio("Select Page", ["Introduction", "IPO Search", "Model Performance", "Investment Strategies", "Feature Analysis"])

st.sidebar.markdown("---")
if metadata:
    st.sidebar.info(f"""
**Total IPOs:** {metadata.get('total_ipos', 'N/A')}
**Test Set:** {metadata.get('test_size', 'N/A')}
**Best Classifier:** {metadata.get('best_classifier_name', 'N/A')}
**Best Regressor:** {metadata.get('best_regressor_name', 'N/A')}
""")

st.sidebar.markdown("---")
st.sidebar.markdown("Logan Wesselt, Julian Tashjian, Dylan Bollinger")

if page == "Introduction":
    st.title("Machine Learning for IPO Risk Prediction")
    st.markdown("### Real-World IPO Data (2019-2024)")
    st.markdown("---")

    st.markdown("""
    <div class="highlight-box">
    <h3>Research Questions</h3>
    <ol>
    <li><strong>Which pre-IPO characteristics most strongly affect first-day returns?</strong></li>
    <li><strong>Can ML classify "high-risk" IPOs more accurately than baseline heuristics?</strong></li>
    <li><strong>Can ML predictions construct superior investment strategies?</strong></li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="success-box">
        <h4>Real-World Data</h4>
        <ul>
        <li><strong>30 actual US IPOs</strong> (2019-2024)</li>
        <li>Snowflake, DoorDash, Airbnb, Coinbase, Rivian, Reddit, ARM</li>
        <li>Real market data from Yahoo Finance</li>
        <li>Actual first-day returns</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="warning-box">
        <h4>Important Limitations</h4>
        <ul>
        <li>Limited to 30 well-known IPOs</li>
        <li>Selection bias toward successful companies</li>
        <li>Educational purpose only</li>
        <li>Not investment advice</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

elif page == "IPO Search":
    st.title("IPO Search & Prediction Tool")
    st.markdown("### Look up real IPOs and view ML predictions")
    st.markdown("---")

    search_term = st.text_input("Enter company name or ticker:", placeholder="e.g., Snowflake, SNOW")

    if st.button("Search", type="primary"):
        if search_term:
            search_term = search_term.upper().strip()
            result = test_preds[test_preds['ticker'].str.upper() == search_term]

            if result.empty:
                result = test_preds[test_preds['company_name'].str.upper().str.contains(search_term, na=False)]

            if not result.empty:
                ipo = result.iloc[0]
                st.success(f"Found: **{ipo['company_name']}** ({ipo['ticker']})")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Actual First-Day Return", f"{ipo['first_day_return'] * 100:.2f}%")

                with col2:
                    st.metric("Predicted Return", f"{ipo['predicted_return'] * 100:.2f}%")

                with col3:
                    risk_label = get_risk_label(ipo['predicted_risk_prob'])
                    st.metric("Risk Classification", risk_label)

                with col4:
                    correct = "Correct" if ipo['predicted_high_risk'] == ipo['high_risk_ipo'] else "Incorrect"
                    st.metric("Prediction Accuracy", correct)

                st.markdown("---")
                st.markdown("#### Key Characteristics")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown(f"**Offer Price:** ${ipo.get('offer_price', 'N/A')}")
                    st.markdown(f"**Industry:** {ipo.get('industry', 'N/A')}")

                with col2:
                    st.markdown(f"**Firm Age:** {ipo.get('firm_age', 'N/A')} years")
                    st.markdown(f"**VIX at IPO:** {ipo.get('vix_level', 'N/A'):.1f}")

                with col3:
                    vc_status = "Yes" if ipo.get('vc_backed', 0) == 1 else "No"
                    prof_status = "Yes" if ipo.get('is_profitable', 0) == 1 else "No"
                    st.markdown(f"**VC-Backed:** {vc_status}")
                    st.markdown(f"**Profitable:** {prof_status}")
            else:
                st.warning(f"No IPO found matching '{search_term}'")

    st.markdown("---")
    st.markdown("### Browse All IPOs")

    if test_preds is not None:
        display_cols = ['company_name', 'ticker', 'industry', 'first_day_return', 'predicted_return']
        display_df = test_preds[[col for col in display_cols if col in test_preds.columns]].copy()

        for col in ['first_day_return', 'predicted_return']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x * 100:.2f}%")

        st.dataframe(display_df, use_container_width=True, hide_index=True)
elif page == "Model Performance":
    st.title("Model Performance Analysis")
    st.markdown("### How well do ML models predict IPO risk?")
    st.markdown("---")

    st.subheader("Classification Performance (High-Risk Prediction)")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("#### ML Model Comparison")
        if clf_results is not None:
            st.dataframe(clf_results, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("#### Baseline Performance")
        st.markdown("""
        **Most Frequent:** 0.500 AUC

        **VIX Rule:** ~0.550 AUC

        **Young + Unprofitable:** ~60% Accuracy
        """)

    if clf_results is not None:
        best_auc = clf_results['Test_AUC'].max()
        best_model = clf_results.loc[clf_results['Test_AUC'].idxmax(), 'Model']

        st.success(f"""
        **Best Model:** {best_model}

        - **AUC:** {best_auc:.3f}
        - **Improvement:** Significantly better than baselines
        """)

    st.markdown("---")
    st.subheader("Regression Performance (Return Prediction)")

    if reg_results is not None:
        st.dataframe(reg_results, use_container_width=True, hide_index=True)

        best_rmse = reg_results['Test_RMSE'].min()
        best_model = reg_results.loc[reg_results['Test_RMSE'].idxmin(), 'Model']

        st.success(f"""
        **Best Model:** {best_model}

        - **RMSE:** {best_rmse:.4f}
        - **Significantly better than mean baseline**
        """)

    st.markdown("---")
    st.subheader("Actual vs. Predicted Returns")

    if test_preds is not None:
        fig = px.scatter(
            test_preds,
            x='first_day_return',
            y='predicted_return',
            color='high_risk_ipo',
            hover_data=['company_name', 'ticker'],
            labels={'first_day_return': 'Actual Return', 'predicted_return': 'Predicted Return'},
            title='Actual vs. Predicted First-Day Returns'
        )

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

elif page == "Investment Strategies":
    st.title("Investment Strategy Evaluation")
    st.markdown("### Can ML predictions construct superior strategies?")
    st.markdown("---")

    st.info("""
    We test four investment strategies on a hypothetical $1,000,000 portfolio:

    1. **Naive Strategy:** Invest equally in ALL IPOs
    2. **ML Avoid High-Risk:** Only invest in predicted low-risk IPOs
    3. **ML Top Quartile:** Only invest in top 25% predicted returns
    4. **ML Combined:** Top quartile AND low-risk (most selective)
    """)

    if strategy_results is not None:
        st.markdown("---")
        st.subheader("Strategy Performance Comparison")
        st.dataframe(strategy_results, use_container_width=True, hide_index=True)

        best_idx = strategy_results['Mean Return (%)'].idxmax()
        best_strategy = strategy_results.loc[best_idx, 'Strategy']
        best_return = strategy_results.loc[best_idx, 'Mean Return (%)']
        naive_return = strategy_results[strategy_results['Strategy'] == 'Naive']['Mean Return (%)'].values[0]
        improvement = best_return - naive_return

        st.success(f"""
        **Best Strategy:** {best_strategy}

        - **Mean Return:** {best_return:.2f}%
        - **vs. Naive:** +{improvement:.2f}%
        - **Dollar Impact:** ${improvement * 10000:,.0f} on $1M portfolio
        """)

        fig = px.bar(
            strategy_results,
            x='Strategy',
            y='Mean Return (%)',
            title='Strategy Performance Comparison',
            color='Mean Return (%)',
            color_continuous_scale='RdYlGn'
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
elif page == "Feature Analysis":
    st.title("Feature Importance Analysis")
    st.markdown("### Which characteristics most affect first-day returns?")
    st.markdown("---")

    if feature_importance is not None:
        st.subheader("Top 10 Most Predictive Features")

        top_10 = feature_importance.head(10)

        fig = px.bar(
            top_10,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Feature Importance (SHAP Values)',
            color='Importance',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Top 3 Features Explained")

        for idx, (_, row) in enumerate(top_10.head(3).iterrows(), 1):
            st.markdown(f"""
            <div class="success-box">
            <strong>#{idx}: {row['Feature']}</strong><br>
            <em>Importance: {row['Importance']:.4f}</em>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### All Features Ranked")
        st.dataframe(feature_importance, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("""
        <div class="highlight-box">
        <h4>Key Insights</h4>
        <p>The top predictive factors are typically:</p>
        <ul>
        <li><strong>Market Conditions:</strong> VIX level, S&P 500 momentum</li>
        <li><strong>Deal Structure:</strong> Offer size, gross proceeds</li>
        <li><strong>Firm Characteristics:</strong> Age, profitability, VC backing</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("**IPO Risk Prediction Dashboard** | JLD Inc. LLC. Partners | FIN 377 Final Project")