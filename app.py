import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pyreadr

# Scikit-Learn Imports
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, precision_recall_curve

# Model Imports
from xgboost import XGBClassifier

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="Deliveroo AI Suite", page_icon="🚲", layout="wide")

# Custom CSS for a Professional Look
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        div[data-testid="stMetricValue"] {font-size: 24px; font-weight: 700;} 
        .stTabs [aria-selected="true"] {background-color: #00CCBC !important; color: white !important;}
        div[data-testid="stDataFrame"] {width: 100%;}
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_rdata(file_bytes: bytes):
    # pyreadr can read from a file path OR a file-like object.
    # We'll write bytes to a temp file to be safe across environments.
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".RData", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    result = pyreadr.read_r(tmp_path)
    return result

# DATA LOADER

st.sidebar.header("📦 Data")

uploaded_rdata = st.sidebar.file_uploader("Upload DeliveryAdClick.RData", type=["rdata", "RData"])

use_local = st.sidebar.checkbox("Use local file in project folder", value=True)

result = None

if uploaded_rdata is not None:
    # Load from uploaded file
    try:
        result = load_rdata(uploaded_rdata.getvalue())
        st.sidebar.success("Loaded RData from upload ✅")
    except Exception as e:
        st.sidebar.error(f"Failed to read uploaded RData: {e}")
        st.stop()

elif use_local:
    # Load from local file
    try:
        result = pyreadr.read_r("DeliveryAdClick.RData")
        st.sidebar.success("Loaded local DeliveryAdClick.RData ✅")
    except FileNotFoundError:
        st.sidebar.error("🚨 'DeliveryAdClick.RData' not found in this folder.")
        st.stop()
    except Exception as e:
        st.sidebar.error(f"Failed to read local RData: {e}")
        st.stop()
else:
    st.info("Upload an .RData file or enable local loading in the sidebar.")
    st.stop()

# Extract datasets
if "ClickTraining" not in result.keys() or "ClickPrediction" not in result.keys():
    st.error(f"Expected objects 'ClickTraining' and 'ClickPrediction' in RData, found: {list(result.keys())}")
    st.stop()

df_train = result["ClickTraining"].copy()
df_new = result["ClickPrediction"].copy()

# Store for other tabs
st.session_state["df_train"] = df_train
st.session_state["df_new"] = df_new

st.sidebar.write(f"Train rows: {df_train.shape[0]:,} | cols: {df_train.shape[1]:,}")
st.sidebar.write(f"Pred rows: {df_new.shape[0]:,} | cols: {df_new.shape[1]:,}")

TARGET_COL = "Clicks_Conversion"  # <- keep your original for now

if TARGET_COL not in df_train.columns:
    st.error(f"Target column '{TARGET_COL}' not found in training data. Columns: {list(df_train.columns)[:20]} ...")
    st.stop()

y = df_train[TARGET_COL].astype(int)
X = df_train.drop(columns=[TARGET_COL])

st.session_state["X_train_full"] = X
st.session_state["y_train_full"] = y

with st.expander("🔍 Preview datasets"):
    st.write("Training dataset head:")
    st.dataframe(df_train.head())
    st.write("Prediction dataset head:")
    st.dataframe(df_new.head())


from sklearn.model_selection import GridSearchCV, StratifiedKFold

# -----------------------------
# 1. UPDATED MODEL PIPELINE
# -----------------------------
def build_xgb_pipeline(X):
    # Identify numeric vs categorical columns
    num_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    # Preprocessing for Numerical Data
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    # Preprocessing for Categorical Data
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    # Bundle preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        verbose_feature_names_out=False
    )

    # Base Model - Parameters will be set by GridSearchCV
    # We name this step 'classifier' to match your param dictionary
    model = XGBClassifier(
        eval_metric='logloss', 
        random_state=42,
        n_jobs=-1
    )

    # Full Pipeline
    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("classifier", model) # <--- Named 'classifier' to match params
    ])
    
    return pipe

# -----------------------------
# 2. TRAINING LOGIC (Grid Search)
# -----------------------------
def train_xgb_and_store(X: pd.DataFrame, y: pd.Series, val_size=0.2):
    """
    Trains XGBoost using GridSearchCV with the specific params provided.
    Stores results in session_state.
    """
    
    # 1. Split Data (Train vs Hold-out Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=val_size, random_state=42, stratify=y
    )

    # 2. Define Pipeline
    pipe = build_xgb_pipeline(X_train)

    # 3. Your Specific Parameters
    param_grid = {
        'classifier__learning_rate': [0.1],
        'classifier__n_estimators': [100],
        'classifier__max_depth': [3]
    }

    # 4. Run Grid Search (Even with 1 param set, this is useful for Cross Validation)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    grid = GridSearchCV(
        pipe, 
        param_grid=param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )

    with st.spinner("⚡ Training XGBoost Model..."):
        grid.fit(X_train, y_train)

    # 5. Get Best Model
    best_model = grid.best_estimator_
    
    # 6. Evaluate on the Hold-out Test Set
    y_prob = best_model.predict_proba(X_test)[:, 1]
    y_pred = best_model.predict(X_test)

    metrics = {
        "AUC": roc_auc_score(y_test, y_prob),
        "f1_score": f1_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "Accuracy": accuracy_score(y_test, y_pred),
    }

    # 7. Store everything in Session State
    st.session_state["model"] = best_model
    st.session_state["metrics"] = metrics
    st.session_state["X_test"] = X_test
    st.session_state["y_test"] = y_test
    st.session_state["y_pred"] = y_pred
    st.session_state["y_prob"] = y_prob

    return best_model, metrics

# -----------------------------
# 3. SIDEBAR: MODEL CONFIGURATION
# -----------------------------
st.sidebar.divider()
st.sidebar.header("🤖 Model Configuration")

# 1. Validation Split Slider
# Controls how much data is held back to test accuracy
val_size = st.sidebar.slider(
    "Validation Split Size", 
    min_value=0.1, 
    max_value=0.4, 
    value=0.2, 
    help="Percentage of data reserved for testing the model."
)

# 2. Decision Threshold Slider
# This variable is used by Tab 3 & Tab 4 to decide what counts as a "Click"
threshold = st.sidebar.slider(
    "Decision Threshold", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.5, 
    step=0.05,
    help="Probability cutoff. Users above this score are predicted as 'Click'."
)

# 3. Train Button
if st.sidebar.button("Train Model", type="primary"):
    if 'df_train' in st.session_state:
        # Load data from session
        X = st.session_state["X_train_full"]
        y = st.session_state["y_train_full"]
        
        with st.spinner("Training XGBoost Model..."):
            # Pass the val_size from the slider into the training function
            model, metrics = train_xgb_and_store(X, y, val_size=val_size)
            
            st.sidebar.success("Training Complete!")
            st.sidebar.write(f"**AUC Score:** {metrics['AUC']:.3f}")
    else:
        st.sidebar.error("Please upload data first.")

# -----------------------------
# MAIN TABS SETUP
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Model Metrics", 
    "💼 Business Insights", 
    "👤 Single Prediction", 
    "📂 Batch Predictions"
])

# -----------------------------
# TAB 1: MODEL PERFORMANCE METRICS
# -----------------------------
with tab1:
    st.header("📊 Model Performance Evaluation")

    if "metrics" not in st.session_state:
        st.info("👆 Please go to the sidebar and click **'Train Model'** to generate results.")
    
    else:
        # Retrieve data
        metrics = st.session_state["metrics"]
        model = st.session_state["model"]
        y_test = st.session_state["y_test"]
        y_pred = st.session_state["y_pred"]
        y_prob = st.session_state["y_prob"]

        # --- ROW 1: METRICS ---
        c1, c2, c3, c4 , c5 = st.columns(5)
        c1.metric("ROC AUC Score", f"{metrics['AUC']:.3f}")
        c2.metric("F1 Score", f"{metrics['f1_score']:.2%}")
        c3.metric("Precision", f"{metrics['Precision']:.2%}")
        c4.metric("Recall", f"{metrics['Recall']:.2%}")
        c5.metric("Accuracy", f"{metrics['Accuracy']:.2%}")
       
        
        st.divider()

        # --- ROW 2: STANDARD PLOTS ---
        col_viz1, col_viz2 = st.columns(2)

        with col_viz1:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = px.imshow(
                cm, text_auto=True, aspect="auto", color_continuous_scale='Teal',
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=['No Click', 'Click'], y=['No Click', 'Click']
            )
            fig_cm.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_cm, use_container_width=True)

        with col_viz2:
            st.subheader("ROC Curve")
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            fig_roc = px.area(x=fpr, y=tpr, labels=dict(x='False Positive Rate', y='True Positive Rate'))
            fig_roc.add_shape(type='line', line=dict(dash='dash', color='gray'), x0=0, x1=1, y0=0, y1=1)
            fig_roc.update_yaxes(scaleanchor="x", scaleratio=1)
            st.plotly_chart(fig_roc, use_container_width=True)

        st.divider()

        # --- ROW 3: ADVANCED DIAGNOSTICS (NEW!) ---
        col_adv1, col_adv2 = st.columns(2)

        # 1. Precision-Recall Curve (Crucial for Imbalanced Data)
        with col_adv1:
            st.subheader("Precision-Recall Curve")
            st.caption("Better than ROC for rare events (like Ad Clicks).")
            
            prec, rec, _ = precision_recall_curve(y_test, y_prob)
            
            fig_pr = px.area(
                x=rec, y=prec, 
                labels=dict(x='Recall (Finds all clicks?)', y='Precision (Accurate clicks?)'),
                title=f'PR AUC: {average_precision_score(y_test, y_prob):.3f}'
            )
            # Add baseline (percentage of positives in data)
            baseline = y_test.mean()
            fig_pr.add_hline(y=baseline, line_dash="dash", line_color="red", annotation_text="Random Baseline")
            
            st.plotly_chart(fig_pr, use_container_width=True)

        # 2. Prediction Distribution (Histogram)
        with col_adv2:
            st.subheader("Model Confidence Histogram")
            st.caption("Is the model decisive? (Peaks at 0 and 1 are good)")
            
            # Create a dataframe for plotting
            hist_df = pd.DataFrame({'Probability': y_prob, 'Actual Outcome': y_test})
            hist_df['Actual Outcome'] = hist_df['Actual Outcome'].map({0: 'No Click', 1: 'Click'})
            
            fig_hist = px.histogram(
                hist_df, 
                x="Probability", 
                color="Actual Outcome", 
                nbins=50,
                barmode="overlay", # Overlap histograms to see separation
                color_discrete_map={'No Click': 'red', 'Click': 'teal'},
                opacity=0.6
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        # --- ROW 4: FEATURE IMPORTANCE ---
        st.subheader("🔍 Feature Importance")
        try:
            xgb_model = model.named_steps['classifier']
            preprocessor = model.named_steps['preprocess']
            feature_names = preprocessor.get_feature_names_out()
            importances = xgb_model.feature_importances_
            
            feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            feat_df = feat_df.sort_values(by='Importance', ascending=True).tail(10)
            
            fig_imp = px.bar(feat_df, x='Importance', y='Feature', orientation='h', text_auto='.3f', color='Importance', color_continuous_scale='Teal')
            fig_imp.update_layout(showlegend=False)
            st.plotly_chart(fig_imp, use_container_width=True)
        except:
            st.warning("Could not extract feature importance.")


# -----------------------------
# TAB 2: BUSINESS INSIGHTS & KPIs
# -----------------------------
with tab2:
    st.header("💼 Historical Business Insights")
    st.info("ℹ️ **Note:** These insights are based on **Historical Training Data**.")

    if "df_train" not in st.session_state:
        st.warning("⚠️ Please load the 'ClickTraining' dataset in the sidebar first.")
    else:
        df = st.session_state["df_train"]
        target = "Clicks_Conversion"

        # --- ROW 1: KPIs ---
        total_obs = len(df)
        total_conversions = df[target].sum()
        conversion_rate = df[target].mean()
        missed_opps = total_obs - total_conversions
        
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Historical Conversion Rate", f"{conversion_rate:.1%}")
        kpi2.metric("Total Historical Users", f"{total_obs:,}")
        kpi3.metric("Missed Opportunities", f"{missed_opps:,}")
        kpi4.metric("Avg Prev. Orders", f"{df['Number_of_Previous_Orders'].mean():.1f}")
        
        st.divider()

        # --- ROW 2: CATEGORICAL DRIVERS ---
        st.subheader("1. Segmentation: Who are we targeting?")
        
        c1, c2, c3 = st.columns(3)
        
        # 1. Social Network
        with c1:
            st.markdown("**By Social Network**")
            social_conv = df.groupby("Social_Network")[target].mean().reset_index()
            fig_social = px.bar(social_conv, x="Social_Network", y=target, color=target, color_continuous_scale='RdYlGn', range_y=[0, 1])
            fig_social.add_hline(y=conversion_rate, line_dash="dash", line_color="black")
            fig_social.update_layout(yaxis_tickformat='.0%')
            st.plotly_chart(fig_social, use_container_width=True)

        # 2. Region
        with c2:
            st.markdown("**By Region**")
            region_conv = df.groupby("Region")[target].mean().reset_index().sort_values(by=target)
            fig_region = px.bar(region_conv, x=target, y="Region", orientation='h', color=target, color_continuous_scale='RdYlGn', range_x=[0, 1])
            fig_region.add_vline(x=conversion_rate, line_dash="dash", line_color="black")
            fig_region.update_layout(xaxis_tickformat='.0%')
            st.plotly_chart(fig_region, use_container_width=True)

        # 3. Carrier (Red-Yellow-Green)
        with c3:
            st.markdown("**By Mobile Carrier**")
            carrier_conv = df.groupby("Carrier")[target].mean().reset_index().sort_values(by=target)
            fig_carrier = px.bar(
                carrier_conv, 
                x="Carrier", 
                y=target, 
                color=target, 
                color_continuous_scale='RdYlGn', 
                range_y=[0, 1]
            )
            fig_carrier.add_hline(y=conversion_rate, line_dash="dash", line_color="black")
            fig_carrier.update_layout(yaxis_tickformat='.0%')
            st.plotly_chart(fig_carrier, use_container_width=True)

        st.divider()

        # --- ROW 3: TEMPORAL PATTERNS ---
        st.subheader("2. Timing: When do they click?")
        
        c4, c5 = st.columns(2)
        
        # 4. Weekday Analysis
        with c4:
            st.markdown("**Conversion by Weekday**")
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            weekday_conv = df.groupby("Weekday")[target].mean().reindex(day_order).reset_index()
            fig_week = px.bar(weekday_conv, x="Weekday", y=target, color=target, color_continuous_scale='Viridis', range_y=[0, 1])
            fig_week.update_layout(yaxis_tickformat='.0%')
            st.plotly_chart(fig_week, use_container_width=True)

        # 5. Daytime Distribution
        with c5:
            st.markdown("**Impact of Time of Day**")
            df_chart = df.copy()
            df_chart['Status'] = df_chart[target].map({0: 'No Click', 1: 'Click'})
            fig_daytime = px.box(
                df_chart, x='Status', y='Daytime', color='Status',
                color_discrete_map={'No Click': '#EF553B', 'Click': '#00CCBC'}
            )
            st.plotly_chart(fig_daytime, use_container_width=True)
            
        st.divider()

        # --- ROW 4: BEHAVIORAL ---
        st.subheader("3. User Behavior")
        
        c6, c7 = st.columns(2)

        # 6. Time on Website
        with c6:
            st.markdown("**Time on Previous Website**")
            fig_time = px.box(
                df_chart, x='Status', y='Time_On_Previous_Website', color='Status',
                color_discrete_map={'No Click': '#EF553B', 'Click': '#00CCBC'}, points=False
            )
            st.plotly_chart(fig_time, use_container_width=True)
            
        # 7. Restaurant Type (FIXED COLOR SCALE)
        with c7:
            st.markdown("**Interest by Restaurant Type**")
            rest_conv = df.groupby("Restaurant_Type")[target].mean().reset_index().sort_values(by=target)
            
            fig_rest = px.bar(
                rest_conv, 
                x="Restaurant_Type", 
                y=target, 
                color=target, 
                color_continuous_scale='RdYlGn' # <--- Changed from 'Purples' to 'RdYlGn'
            )
            fig_rest.update_layout(yaxis_tickformat='.0%')
            st.plotly_chart(fig_rest, use_container_width=True)
    st.divider()
    st.subheader("4. Conversion Rate by Customer Loyalty")
    st.caption(
        "Here we analyze how conversion likelihood changes as users place more previous orders. "
        "This would help identify high-value repeat customers."
    )
    # Cap extreme values to avoid long tails (optional but recommended)
    df_loyalty = df.copy()
    df_loyalty["Orders_Bucket"] = pd.cut(
        df_loyalty["Number_of_Previous_Orders"],
        bins=[-1, 0, 1, 3, 5, 10, np.inf],
        labels=[
            "0 (New)",
            "1",
            "2–3",
            "4–5",
            "6–10",
            "10+"
        ]
    )

    loyalty_conv = (
        df_loyalty
        .groupby("Orders_Bucket")[target]
        .mean()
        .reset_index()
    )
    fig_loyalty = px.bar(
        loyalty_conv,
        x="Orders_Bucket",
        y=target,
        color=target,
        color_continuous_scale="RdYlGn",
        range_y=[0, 1],
        labels={
            "Orders_Bucket": "Number of Previous Orders",
            target: "Conversion Rate"
        },
        text=loyalty_conv[target].apply(lambda x: f"{x:.1%}")
    )

    fig_loyalty.add_hline(
        y=conversion_rate,
        line_dash="dash",
        line_color="black",
        annotation_text="Overall Avg"
    )

    fig_loyalty.update_layout(
        yaxis_tickformat=".0%",
        showlegend=False
    )

    st.plotly_chart(fig_loyalty, use_container_width=True)
    best_bucket = loyalty_conv.loc[loyalty_conv[target].idxmax(), "Orders_Bucket"]
    best_rate = loyalty_conv[target].max()

    st.success(
        f"💡 Users with **{best_bucket} previous orders** show the highest conversion rate "
        f"(**{best_rate:.1%}**), indicating strong loyalty effects."
    )
    st.divider()
    st.subheader("📍 Geographic Disparity: Conversion Rate by Region")
    st.caption(
        "Highlights regional differences in conversion performance, "
        "helping identify high- and low-performing markets."
    )

    region_conv = (
        df.groupby("Region")[target]
        .mean()
        .reset_index()
        .sort_values(by=target)
    )

    fig_geo = px.bar(
        region_conv,
        x=target,
        y="Region",
        orientation="h",
        color=target,
        color_continuous_scale="RdYlGn",
        labels={target: "Conversion Rate"},
        text=region_conv[target].apply(lambda x: f"{x:.1%}")
    )

    fig_geo.add_vline(
        x=conversion_rate,
        line_dash="dash",
        line_color="black",
        annotation_text="Overall Avg"
    )

    fig_geo.update_layout(
        xaxis_tickformat=".0%",
        height=450,
        showlegend=False
    )

    st.plotly_chart(fig_geo, use_container_width=True)
    best_region = region_conv.iloc[-1]["Region"]
    worst_region = region_conv.iloc[0]["Region"]

    st.success(
        f"💡 Conversion rates are broadly consistent across regions, with modest differences. Hence geographic level targetting is not necessary"
       
    )









# -----------------------------
# TAB 3: SINGLE PREDICTION
# -----------------------------
with tab3:
    st.header("👤 Real-Time Prediction Simulator")
    st.info("Adjust the profile below. The AI will predict the probability of a click based on historical patterns.")
    
    # 1. Check if model & data exist
    if "model" not in st.session_state or "df_train" not in st.session_state:
        st.warning("⚠️ Please train the model in the sidebar first!")
    else:
        model = st.session_state["model"]
        df_train = st.session_state["df_train"]
        
        # --- INPUT FORM ---
        with st.form("prediction_form"):
            st.subheader("Define User Profile")
            
            c1, c2, c3 = st.columns(3)
            
            # Column 1
            with c1:
                st.markdown("**📍 Location & Device**")
                # Drop N/A values to prevent sorting errors
                region = st.selectbox("Region", options=sorted(df_train['Region'].dropna().unique()))
                carrier = st.selectbox("Mobile Carrier", options=sorted(df_train['Carrier'].dropna().unique()))
                
            # Column 2
            with c2:
                st.markdown("**⏰ Context**")
                weekday = st.selectbox("Weekday", options=sorted(df_train['Weekday'].dropna().unique()))
                social = st.selectbox("Social Network", options=sorted(df_train['Social_Network'].dropna().unique()))
                daytime = st.slider("Time of Day (0=Morning, 1=Night)", 0.0, 1.0, 0.5)
                
            # Column 3
            with c3:
                st.markdown("**🍔 User History**")
                rest_type = st.selectbox("Restaurant Interest", options=sorted(df_train['Restaurant_Type'].dropna().unique()))
                
                default_time = float(df_train['Time_On_Previous_Website'].mean())
                time_web = st.number_input("Time on Prev. Website (sec)", min_value=0.0, value=default_time)
                
                default_orders = int(df_train['Number_of_Previous_Orders'].median())
                prev_orders = st.number_input("Previous Orders", min_value=0, step=1, value=default_orders)
            
            st.markdown("---")
            submit_btn = st.form_submit_button("🔮 Predict Probability", type="primary")

        # --- PREDICTION RESULT (Outside the form) ---
        if submit_btn:
            # Prepare Input Data
            input_data = pd.DataFrame({
                'Region': [region],
                'Daytime': [daytime],
                'Carrier': [carrier],
                'Time_On_Previous_Website': [time_web],
                'Weekday': [weekday],
                'Social_Network': [social],
                'Number_of_Previous_Orders': [prev_orders],
                'Restaurant_Type': [rest_type]
            })
            
            # Get Prediction
            prob = model.predict_proba(input_data)[0][1]
            
            # --- USE SIDEBAR THRESHOLD INSTEAD OF 0.5 ---
            # 'threshold' variable comes from the slider in your sidebar
            is_click = prob > threshold 
            
            st.subheader("Prediction Results")
            
            col_res1, col_res2 = st.columns([1, 2])
            
            with col_res1:
                if is_click:
                    st.success(f"## ✅ LIKELY CLICK\n**Probability: {prob:.1%}**")
                    st.caption(f"Reason: Probability ({prob:.2f}) is higher than the threshold ({threshold:.2f}).")
                else:
                    st.error(f"## ❌ NO CLICK\n**Probability: {prob:.1%}**")
                    st.caption(f"Reason: Probability ({prob:.2f}) is lower than the threshold ({threshold:.2f}).")

            with col_res2:
                # Gauge Chart
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prob * 100,
                    title = {'text': f"Conversion Probability (Threshold: {threshold:.0%})"},
                    number ={
                    "suffix": "%",
                    "font": {
                        "size": 40,        # ⬅️ smaller number (try 36–44)
                        "color": "white"
                    },
                    "valueformat": ".1f"
                }
,
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#00CCBC" if is_click else "#EF553B"},
                        'steps': [
                            {'range': [0, threshold * 100], 'color': "lightgray"},
                            {'range': [threshold * 100, 100], 'color': "white"}],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': threshold * 100} # Visual line at the threshold
                    }
                ))
                fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=10))
                st.plotly_chart(fig_gauge, use_container_width=True)


# -----------------------------
# TAB 4: BATCH PREDICTION & ANALYTICS
# -----------------------------
with tab4:
    st.header("📂 Batch Prediction & Analytics")
    st.info("Analyze the 2,000 unlabelled points in 'ClickPrediction' to forecast their behavior.")

    if "model" not in st.session_state:
        st.warning("⚠️ Please train the model in the sidebar first.")
    elif "df_new" not in st.session_state:
        st.warning("⚠️ Please load the 'ClickPrediction' data in the sidebar.")
    else:
        model = st.session_state["model"]
        df_new = st.session_state["df_new"].copy() 
        
        st.write(f"**Loaded Prediction Dataset:** {df_new.shape[0]:,} rows")
        
        if st.button("🚀 Run Predictions & Analyze Batch", type="primary"):
            
            with st.spinner("Generating predictions..."):
                # 1. Calculate Probabilities
                probs = model.predict_proba(df_new)[:, 1]
                
                # Append to Dataframe
                df_new['Click_Probability'] = probs
                
                # We still calculate "Likely Click" for the Summary KPIs (using threshold)
                df_new['Predicted_Status'] = (probs > threshold).astype(int)
                df_new['Status_Label'] = df_new['Predicted_Status'].map({1: 'Likely Click', 0: 'No Click'})
                
                # --- SECTION A: SUMMARY KPIs ---
                st.subheader("1. Batch Summary Statistics")
                
                total_users = len(df_new)
                avg_confidence = probs.mean()
                # "Expected Conversions" is the sum of all probabilities (Statistical Expected Value)
                expected_conversions = probs.sum()
                
                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                kpi1.metric("Batch Size", f"{total_users:,}")
                kpi2.metric("Avg. Click Probability", f"{avg_confidence:.1%}", help="The average likelihood of a click across the entire file.")
                kpi3.metric("Expected Conversions", f"{expected_conversions:.0f}", help="Sum of all probabilities. statistically, this is how many clicks you get.")
                kpi4.metric("Threshold Selected", f"{threshold:.0%}")
                
                st.divider()
                
                # --- SECTION B: PROBABILITY DISTRIBUTION ---
                st.subheader("2. Probability Distribution")
                st.caption(f"How 'sure' is the model about these 2,000 users?")
                fig_hist = px.histogram(
                    df_new, 
                    x="Click_Probability", 
                    nbins=50,
                    color="Status_Label",
                    color_discrete_map={'Likely Click': '#00CCBC', 'No Click': '#EF553B'},
                    range_x=[0, 1]
                )
                fig_hist.add_vline(x=threshold, line_dash="dash", line_color="black", annotation_text="Threshold")
                st.plotly_chart(fig_hist, use_container_width=True)
                
                st.divider()
                
                # --- SECTION C: QUALITY SEGMENTATION (AVERAGE PROBABILITY) ---
                st.subheader("3. Segmentation by Quality (Avg. Probability)")
                st.caption("Which groups have the highest **propensity** to click? (Higher % is better)")

                c1, c2 = st.columns(2)
                
                # Chart 1: Avg Prob by Region
                with c1:
                    st.markdown("**Region Quality**")
                    # Calculate Mean Probability per Region
                    reg_qual = df_new.groupby('Region')['Click_Probability'].mean().reset_index()
                    fig_reg = px.bar(
                        reg_qual, 
                        x='Click_Probability', 
                        y='Region', 
                        orientation='h', 
                        color='Click_Probability', 
                        color_continuous_scale='Teal',
                        labels={'Click_Probability': 'Avg Probability'},
                        range_x=[0, 1]
                    )
                    fig_reg.update_layout(xaxis_tickformat='.0%')
                    st.plotly_chart(fig_reg, use_container_width=True)

                # Chart 2: Avg Prob by Social Network
                with c2:
                    st.markdown("**Social Network Quality**")
                    soc_qual = df_new.groupby('Social_Network')['Click_Probability'].mean().reset_index()
                    fig_soc = px.bar(
                        soc_qual, 
                        x='Social_Network', 
                        y='Click_Probability', 
                        color='Click_Probability', 
                        color_continuous_scale='Teal',
                        labels={'Click_Probability': 'Avg Probability'},
                        range_y=[0, 1]
                    )
                    fig_soc.update_layout(yaxis_tickformat='.0%')
                    st.plotly_chart(fig_soc, use_container_width=True)
                    
                c3, c4 = st.columns(2)
                
                # Chart 3: Avg Prob by Carrier
                with c3:
                    st.markdown("**Carrier Quality**")
                    car_qual = df_new.groupby('Carrier')['Click_Probability'].mean().reset_index()
                    fig_car = px.bar(
                        car_qual, 
                        x='Carrier', 
                        y='Click_Probability', 
                        color='Click_Probability', 
                        color_continuous_scale='Teal',
                        labels={'Click_Probability': 'Avg Probability'},
                        range_y=[0, 1]
                    )
                    fig_car.update_layout(yaxis_tickformat='.0%')
                    st.plotly_chart(fig_car, use_container_width=True)

                # Chart 4: Avg Prob by Restaurant Type
                with c4:
                    st.markdown("**Restaurant Type Quality**")
                    rest_qual = df_new.groupby('Restaurant_Type')['Click_Probability'].mean().reset_index()
                    fig_rest = px.bar(
                        rest_qual, 
                        x='Click_Probability', 
                        y='Restaurant_Type', 
                        orientation='h', 
                        color='Click_Probability', 
                        color_continuous_scale='Teal',
                        labels={'Click_Probability': 'Avg Probability'},
                        range_x=[0, 1]
                    )
                    fig_rest.update_layout(xaxis_tickformat='.0%')
                    st.plotly_chart(fig_rest, use_container_width=True)

                # --- SECTION D: DOWNLOAD ---
                st.divider()
                st.subheader("4. Export Results")
                
                st.write("**Top 5 Highest Potential Users:**")
                top_leads = df_new.sort_values(by="Click_Probability", ascending=False).head(5)
                st.dataframe(top_leads.style.format({'Click_Probability': "{:.1%}"}))
                
                csv = df_new.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Full Predictions CSV",
                    data=csv,
                    file_name="deliveroo_batch_predictions.csv",
                    mime="text/csv"
                )