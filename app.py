"""
TANZANIAN CUSTOMER CHOICE PREDICTOR
Machine Learning Project Test 2
Group [Your Group Number]
February 2026

This application predicts customer satisfaction based on product attributes
and customer demographics in the Tanzanian market.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Tanzania Customer Choice Predictor",
    page_icon="🇹🇿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS FOR BETTER UI
# ============================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0;
        padding-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4B5563;
        text-align: center;
        margin-top: 0;
        padding-top: 0;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border-left: 5px solid #1E3A8A;
    }
    .metric-card {
        background-color: white;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1rem;
        background-color: #F9FAFB;
        border-radius: 10px;
        font-size: 0.9rem;
        color: #6B7280;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# HEADER SECTION
# ============================================
st.markdown('<h1 class="main-header">🇹🇿 Tanzanian Customer Choice Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Machine Learning Model for Predicting Customer Satisfaction in Tanzanian Markets</p>', unsafe_allow_html=True)

# ============================================
# LOAD MODELS AND PREPROCESSING OBJECTS
# ============================================
@st.cache_resource
def load_models():
    """Load all trained models and preprocessing objects"""
    try:
        # Check if files exist
        required_files = [
            'best_model.pkl', 'scaler.pkl', 'feature_names.pkl',
            'label_encoder_region.pkl', 'label_encoder_gender.pkl', 
            'label_encoder_religion.pkl'
        ]
        
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            st.error(f" Missing files: {missing_files}")
            st.info("Please ensure all model files are in the correct directory.")
            return None, None, None, None, None, None
        
        # Load all objects
        model = joblib.load('best_model.pkl')
        scaler = joblib.load('scaler.pkl')
        feature_names = joblib.load('feature_names.pkl')
        region_encoder = joblib.load('label_encoder_region.pkl')
        gender_encoder = joblib.load('label_encoder_gender.pkl')
        religion_encoder = joblib.load('label_encoder_religion.pkl')
        
        return model, scaler, feature_names, region_encoder, gender_encoder, religion_encoder
    
    except Exception as e:
        st.error(f" Error loading models: {str(e)}")
        st.info("Please make sure you've run the Jupyter notebook first to generate all model files.")
        return None, None, None, None, None, None

# Load models
model, scaler, feature_names, region_encoder, gender_encoder, religion_encoder = load_models()

# ============================================
# SIDEBAR - INPUT PARAMETERS
# ============================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/tanzania.png", width=80)
    st.title(" Input Parameters")
    st.markdown("---")
    
    # Create tabs in sidebar for better organization
    tab1, tab2, tab3 = st.tabs(["👤 Customer", "📱 Product", "⚙️ Factors"])
    
    with tab1:
        st.subheader("Customer Demographics")
        
        age = st.slider("Age", 15, 75, 28, help="Customer's age in years")
        
        gender = st.selectbox(
            "Gender", 
            ["Male", "Female"],
            help="Customer's gender"
        )
        
        region = st.selectbox(
            "Region", 
            ['Dar es Salaam', 'Arusha', 'Mwanza', 'Mbeya', 'Zanzibar', 'Dodoma', 'Tanga', 'Morogoro'],
            help="Region in Tanzania"
        )
        
        religion = st.selectbox(
            "Religion", 
            ["Christian", "Muslim", "Other"],
            help="Customer's religious affiliation"
        )
        
        tech_savvy = st.slider(
            "Tech Savvy Score", 1, 5, 3,
            help="1 = Low tech familiarity, 5 = High tech familiarity"
        )
        
    with tab2:
        st.subheader("Product Details")
        
        # Product selection
        brands = ['Samsung', 'Tecno', 'Infinix', 'Nokia', 'Apple', 
                  'Huawei', 'Oppo', 'Vivo', 'Itel', 'Xiaomi', 'Local Brand']
        brand = st.selectbox("Brand", brands)
        
        # Brand tier mapping
        brand_tiers = {
            'Samsung': 'Premium', 'Apple': 'Premium', 'Huawei': 'Premium',
            'Tecno': 'Mid-Range', 'Infinix': 'Mid-Range', 'Nokia': 'Mid-Range',
            'Oppo': 'Mid-Range', 'Vivo': 'Mid-Range', 'Xiaomi': 'Mid-Range',
            'Itel': 'Budget', 'Local Brand': 'Budget'
        }
        brand_tier = brand_tiers[brand]
        
        # Show brand tier
        st.info(f"📊 **Brand Tier:** {brand_tier}")
        
        product_status = st.radio(
            "Product Status", 
            ["New", "Used"],
            help="Whether the product is new or used"
        )
        
        # Price based on tier
        if brand_tier == 'Budget':
            price = st.slider("Price (TZS)", 80000, 250000, 150000, step=5000,
                            format="TZS %d", help="Price in Tanzanian Shillings")
        elif brand_tier == 'Mid-Range':
            price = st.slider("Price (TZS)", 250000, 600000, 400000, step=10000,
                            format="TZS %d")
        else:
            price = st.slider("Price (TZS)", 600000, 2500000, 1200000, step=50000,
                            format="TZS %d")
        
        quality_score = st.slider(
            "Quality Score", 1.0, 10.0, 7.0, step=0.1,
            help="Perceived quality of the product (1-10)"
        )
        
    with tab3:
        st.subheader("Choice Factors")
        st.caption("These factors influence customer decisions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            brand_loyalty = st.slider(
                "Brand Loyalty", 0.0, 1.0, 0.5,
                help="How loyal the customer is to brands"
            )
            
            price_sensitivity = st.slider(
                "Price Sensitivity", 0.0, 1.0, 0.6,
                help="How sensitive customer is to price"
            )
            
            quality_sensitivity = st.slider(
                "Quality Sensitivity", 0.0, 1.0, 0.7,
                help="How much customer values quality"
            )
            
        with col2:
            cultural_pref = st.slider(
                "Cultural Preference", 0.0, 1.0, 0.5,
                help="Strength of cultural preferences"
            )
            
            social_influence = st.slider(
                "Social Influence", 0.0, 1.0, 0.5,
                help="Influence from friends/social media"
            )
            
            origin_pref = st.slider(
                "Origin Preference", 0.0, 1.0, 0.6,
                help="Preference for product origin"
            )
    
    st.markdown("---")
    
    # Advanced factors expander
    with st.expander("🔧 Advanced Factors (Optional)"):
        st.caption("Fine-tune the choice factors")
        
        price_factor = st.slider("Price Factor", 0.0, 1.0, 0.7)
        quality_factor = st.slider("Quality Factor", 0.0, 1.0, 0.6)
        brand_preference = st.slider("Brand Preference", 0.0, 1.0, 0.5)
        cultural_compatibility = st.slider("Cultural Compatibility", 0.0, 1.0, 0.8)
        status_factor = 1.0 if product_status == "New" else st.slider("Status Factor", 0.0, 0.8, 0.5)
    
    # Predict button
    predict_button = st.button("🔮 PREDICT CUSTOMER SATISFACTION", type="primary", use_container_width=True)
    
    # Reset button
    if st.button("🔄 Reset All Inputs", use_container_width=True):
        st.experimental_rerun()

# ============================================
# MAIN CONTENT AREA
# ============================================

if model is not None:
    if predict_button:
        # Prepare input data
        input_data = {
            'age': age,
            'tech_savvy_score': tech_savvy,
            'brand_loyalty': brand_loyalty,
            'price_sensitivity': price_sensitivity,
            'quality_sensitivity': quality_sensitivity,
            'cultural_preference_strength': cultural_pref,
            'price_tzs': price,
            'quality_score': quality_score,
            'price_factor': price_factor,
            'quality_factor': quality_factor,
            'brand_preference': brand_preference,
            'cultural_compatibility': cultural_compatibility,
            'social_influence': social_influence,
            'status_factor': status_factor,
            'origin_factor': origin_pref,
            'region_encoded': region_encoder.transform([region])[0],
            'gender_encoded': gender_encoder.transform([gender])[0],
            'religion_encoded': religion_encoder.transform([religion])[0]
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Ensure correct feature order
        input_df = input_df[feature_names]
        
        # Scale features
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        # ============================================
        # DISPLAY PREDICTION RESULTS
        # ============================================
        
        st.markdown("---")
        st.header(" Prediction Results")
        
        # Create three columns for results
        col1, col2, col3 = st.columns([1.5, 1, 1])
        
        with col1:
            # Gauge chart for satisfaction
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prediction * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Customer Satisfaction Score", 'font': {'size': 24}},
                delta={'reference': 50, 'position': "top"},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkblue", 'thickness': 0.3},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 33], 'color': '#FF6B6B'},  # Red for low
                        {'range': [33, 66], 'color': '#FFD93D'},  # Yellow for medium
                        {'range': [66, 100], 'color': '#6BCB77'}  # Green for high
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            
            fig.update_layout(
                height=350,
                margin=dict(l=20, r=20, t=50, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                font={'color': "darkblue", 'family': "Arial"}
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.subheader("📈 Score Breakdown")
            
            # Display score as percentage
            percentage = prediction * 100
            st.markdown(f"### {percentage:.1f}%")
            
            # Interpretation
            if prediction >= 0.7:
                st.success("✅ HIGH SATISFACTION")
                st.markdown("""
                - Very likely to purchase
                - Strong market potential
                - Consider increasing stock
                """)
            elif prediction >= 0.4:
                st.warning("⚠️ MEDIUM SATISFACTION")
                st.markdown("""
                - Moderate interest
                - May need adjustments
                - Test in specific regions
                """)
            else:
                st.error("❌ LOW SATISFACTION")
                st.markdown("""
                - Unlikely to purchase
                - Reconsider product strategy
                - Check pricing/quality
                """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.subheader("🎯 Key Metrics")
            
            # Calculate derived metrics
            value_ratio = quality_score / (price/100000)
            purchase_likelihood = prediction * 100
            
            # Display metrics in cards
            st.metric("Purchase Likelihood", f"{purchase_likelihood:.1f}%", 
                     delta=f"{purchase_likelihood-50:.1f}% vs avg")
            
            st.metric("Value for Money", f"{value_ratio:.2f}",
                     delta="Good" if value_ratio > 0.7 else "Needs Improvement")
            
            st.metric("Target Segment", 
                     "Premium" if price > 600000 else "Mass Market",
                     delta=region)
        
        # ============================================
        # DETAILED ANALYSIS SECTION
        # ============================================
        
        st.markdown("---")
        st.header("🔍 Detailed Market Analysis")
        
        tab1, tab2, tab3 = st.tabs(["Regional Analysis", "Factor Importance", "Optimization Tips"])
        
        with tab1:
            st.subheader("Predicted Satisfaction by Region")
            
            # Simulate regional predictions
            regions_list = ['Dar es Salaam', 'Arusha', 'Mwanza', 'Mbeya', 'Zanzibar']
            region_scores = []
            
            progress_bar = st.progress(0)
            for i, r in enumerate(regions_list):
                temp_data = input_data.copy()
                temp_data['region_encoded'] = region_encoder.transform([r])[0]
                temp_df = pd.DataFrame([temp_data])[feature_names]
                temp_scaled = scaler.transform(temp_df)
                region_score = model.predict(temp_scaled)[0]
                region_scores.append({'Region': r, 'Satisfaction': region_score})
                progress_bar.progress((i + 1) / len(regions_list))
            
            region_df = pd.DataFrame(region_scores).sort_values('Satisfaction', ascending=False)
            
            # Create bar chart
            fig = px.bar(
                region_df, 
                x='Region', 
                y='Satisfaction',
                title='Customer Satisfaction Across Tanzanian Regions',
                color='Satisfaction',
                color_continuous_scale='Viridis',
                text_auto='.2f'
            )
            
            fig.update_layout(
                xaxis_title="Region",
                yaxis_title="Satisfaction Score",
                yaxis_range=[0, 1],
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Best region recommendation
            best_region = region_df.iloc[0]['Region']
            best_score = region_df.iloc[0]['Satisfaction']
            st.success(f"🎯 **Best Region for this Product:** {best_region} (Score: {best_score:.2f})")
        
        with tab2:
            st.subheader("Factor Importance Analysis")
            
            # Calculate factor contributions
            factors = {
                'Price': price_factor * price_sensitivity,
                'Quality': quality_factor * quality_sensitivity,
                'Brand': brand_preference * brand_loyalty,
                'Culture': cultural_compatibility * cultural_pref,
                'Social': social_influence,
                'Origin': origin_pref
            }
            
            factors_df = pd.DataFrame({
                'Factor': list(factors.keys()),
                'Influence': list(factors.values())
            }).sort_values('Influence', ascending=False)
            
            # Create horizontal bar chart
            fig = px.bar(
                factors_df,
                y='Factor',
                x='Influence',
                orientation='h',
                title='What Drives This Customer\'s Choice?',
                color='Influence',
                color_continuous_scale='Blues',
                text_auto='.2f'
            )
            
            fig.update_layout(
                xaxis_title="Influence Score",
                yaxis_title="",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Top factor
            top_factor = factors_df.iloc[0]['Factor']
            top_influence = factors_df.iloc[0]['Influence']
            st.info(f"💡 **Primary Driver:** {top_factor} (Influence: {top_influence:.2f})")
        
        with tab3:
            st.subheader("Product Optimization Suggestions")
            
            suggestions = []
            
            # Generate optimization suggestions
            if prediction < 0.5:
                if price > 500000 and price_sensitivity > 0.6:
                    suggestions.append({
                        'issue': 'Price too high for sensitive customer',
                        'suggestion': 'Consider lowering price by 15-20%',
                        'impact': 'Could increase satisfaction by 15-25%'
                    })
                
                if quality_score < 6:
                    suggestions.append({
                        'issue': 'Low quality perception',
                        'suggestion': 'Improve product quality or highlight quality features',
                        'impact': 'Could increase satisfaction by 20-30%'
                    })
                
                if cultural_compatibility < 0.5:
                    suggestions.append({
                        'issue': 'Poor cultural fit',
                        'suggestion': 'Adapt marketing message for local culture',
                        'impact': 'Could increase satisfaction by 10-20%'
                    })
                
                if brand_preference < 0.4:
                    suggestions.append({
                        'issue': 'Weak brand appeal',
                        'suggestion': 'Strengthen brand presence in this region',
                        'impact': 'Could increase satisfaction by 15-25%'
                    })
            else:
                suggestions.append({
                    'issue': 'Product is well-positioned',
                    'suggestion': 'Consider expanding to other regions',
                    'impact': 'Potential for 30-40% market expansion'
                })
            
            # Display suggestions in a table
            if suggestions:
                suggestions_df = pd.DataFrame(suggestions)
                st.dataframe(suggestions_df, use_container_width=True)
            else:
                st.success("✅ No major issues detected - product is well-optimized!")
        
        # ============================================
        # EXPORT OPTIONS
        # ============================================
        
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Export prediction as CSV
            if st.button("📥 Export Prediction as CSV"):
                result_df = pd.DataFrame([{
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'region': region,
                    'brand': brand,
                    'price_tzs': price,
                    'predicted_satisfaction': prediction,
                    'recommendation': 'High' if prediction >= 0.7 else 'Medium' if prediction >= 0.4 else 'Low'
                }])
                csv = result_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            # Share results
            if st.button("📧 Share Results"):
                st.info("Share this URL with your team: " + st.get_option("browser.gatherUsageStats"))
        
        with col3:
            # New prediction
            if st.button("🆕 New Prediction"):
                st.experimental_rerun()
    
    else:
        # Welcome message when no prediction made
        st.info("👈 **Enter customer and product details in the sidebar and click 'PREDICT' to see results**")
        
        # Show sample visualizations
        st.markdown("---")
        st.subheader("📊 Sample Market Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Sample distribution chart
            sample_data = pd.DataFrame({
                'Region': ['Dar es Salaam', 'Arusha', 'Mwanza', 'Mbeya', 'Zanzibar'],
                'Avg Satisfaction': [0.72, 0.68, 0.65, 0.58, 0.62]
            })
            
            fig = px.bar(sample_data, x='Region', y='Avg Satisfaction', 
                        title='Average Satisfaction by Region (Sample Data)',
                        color='Avg Satisfaction', color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Sample factor importance
            factors = ['Quality', 'Price', 'Culture', 'Brand', 'Social']
            importance = [0.35, 0.28, 0.18, 0.12, 0.07]
            
            fig = px.pie(values=importance, names=factors, 
                        title='What Drives Customer Choice? (Sample)',
                        color_discrete_sequence=px.colors.sequential.Viridis)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div style='background-color: #EFF6FF; padding: 20px; border-radius: 10px; margin-top: 20px;'>
            <h4>📌 How to use this application:</h4>
            <ol>
                <li><strong>Enter customer details</strong> in the sidebar (age, region, etc.)</li>
                <li><strong>Select product attributes</strong> (brand, price, quality)</li>
                <li><strong>Adjust choice factors</strong> to fine-tune predictions</li>
                <li><strong>Click PREDICT</strong> to see customer satisfaction score</li>
                <li><strong>Review detailed analysis</strong> including regional breakdowns</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

else:
    # Error message if models not loaded
    st.error("❌ **Models could not be loaded!**")
    st.warning("""
    Please ensure:
    1. You have run the Jupyter notebook (`ml_project.ipynb`) first
    2. All model files are in the same directory as this app
    3. Required files: best_model.pkl, scaler.pkl, feature_names.pkl, label_encoder_*.pkl
    """)
    
    # Show directory contents for debugging
    if st.checkbox("Show directory contents"):
        files = os.listdir()
        st.write(files)

# ============================================
# FOOTER
# ============================================
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**🇹🇿 Tanzanian Customer Choice Predictor**")
with col2:
    st.markdown("**Group [Your Group Number]**")
with col3:
    st.markdown("**February 2026**")
st.markdown('</div>', unsafe_allow_html=True)

# Add debug information (optional)
if st.checkbox("Show Debug Info", False):
    st.write("### Debug Information")
    st.write(f"Model loaded: {model is not None}")
    st.write(f"Feature names: {feature_names if feature_names else 'Not loaded'}")