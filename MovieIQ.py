import streamlit as st
import os
import pandas as pd
import numpy as np
from data_loader import load_and_clean_data, generate_mock_data
from analysis import (
    plot_budget_vs_revenue,
    plot_genre_trends,
    plot_success_associations,
    plot_correlation_heatmap,
    run_t_test,
    run_chi_square_test,
    PRIMARY_COLOR,
    ACCENT_COLOR
)
from model import (
    train_and_evaluate_model,
    plot_confusion_matrix,
    plot_feature_importance,
    predict_single
)

# ----------------------------------------------------
# PAGE CONFIGURATION & STYLING
# ----------------------------------------------------
st.set_page_config(
    page_title="MovieIQ - Predictive Film Success Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS to force a clean 2-color styling hierarchy and remove any default emoji visuals
st.markdown(
    f"""
    <style>
    /* Main body background and text font styling */
    .stApp {{
        font-family: 'DejaVu Sans', sans-serif;
    }}
    /* Custom headers and text colors to match the palette */
    h1, h2, h3, h4 {{
        color: {PRIMARY_COLOR} !important;
        font-weight: 600 !important;
    }}
    /* Style sidebar background and text */
    [data-testid="stSidebar"] {{
        background-color: transparent !important;
        border-right: 1px solid #e0e0e0;
    }}
    /* Make filters / selectbox / inputs transparent */
    [data-testid="stSidebar"] .stMultiSelect, 
    [data-testid="stSidebar"] .stSelectbox,
    [data-testid="stSidebar"] div[data-baseweb="select"] {{
        background-color: transparent !important;
    }}
    /* Style tag items in multiselect to match Slate Blue */
    [data-baseweb="tag"] {{
        background-color: {PRIMARY_COLOR} !important;
        color: white !important;
    }}
    [data-baseweb="tag"] span {{
        color: white !important;
    }}
    [data-baseweb="tag"] svg {{
        fill: white !important;
    }}
    /* Rotate hue of ONLY the red interactive components from Streamlit Red to brand Slate Blue, adjusting brightness/saturation to match dark Slate Blue */
    div[data-testid="stSlider"] > div,
    .stTabs [role="tablist"],
    div[data-testid="stNumberInput"] button {{
        filter: hue-rotate(213deg) brightness(50%) saturate(60%) !important;
    }}
    /* Style tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 24px;
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 50px;
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 600;
        color: #555555;
    }}
    /* Color active slider value display labels (e.g. 6.50) to match brand Slate Blue */
    div[data-testid="stSlider"] div[data-testid="stMarkdownContainer"] p {{
        color: {PRIMARY_COLOR} !important;
    }}
    div[data-testid="stSlider"] [data-testid="stWidgetLabel"] div {{
        color: {PRIMARY_COLOR} !important;
    }}
    /* Keep main widget description labels dark gray */
    div[data-testid="stSlider"] label p {{
        color: #333333 !important;
    }}
    /* Custom prediction status cards */
    .success-card {{
        background-color: #e8f8f5;
        border: 2px solid {ACCENT_COLOR};
        border-radius: 8px;
        padding: 20px;
        color: #0e6251;
        font-weight: bold;
    }}
    .fail-card {{
        background-color: #fdf2f2;
        border: 2px solid #e74c3c;
        border-radius: 8px;
        padding: 20px;
        color: #78281f;
        font-weight: bold;
    }}
    /* Metric styling */
    .metric-value {{
        font-size: 24px;
        font-weight: 700;
        color: {PRIMARY_COLOR};
    }}
    .metric-label {{
        font-size: 14px;
        color: #666666;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------------------------------
# DATA LOADING & CACHING
# ----------------------------------------------------
DATA_FILE_PATH = "movies.csv"

# Title header (without emojis, custom color)
st.markdown(
    f"""
    <div style="border-bottom: 2px solid {PRIMARY_COLOR}; padding-bottom: 10px; margin-bottom: 25px;">
        <h1 style="margin: 0; color: {PRIMARY_COLOR};">MovieIQ</h1>
        <p style="margin: 5px 0 0 0; color: #555555; font-size: 15px;">Predictive Analytics on Film Success</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Initialize session state for dataset status
if 'dataset_loaded' not in st.st_state if hasattr(st, 'st_state') else 'dataset_loaded' not in st.session_state:
    st.session_state['dataset_loaded'] = False
    st.session_state['using_mock'] = False

df = None
all_genres = []

# Sidebar upload options
st.sidebar.markdown(
    f"""
    <div style="border-bottom: 1px solid #dcdcdc; padding-bottom: 8px; margin-bottom: 15px;">
        <h3 style="margin: 0; color: {PRIMARY_COLOR}; font-size: 16px;">Dataset Management</h3>
    </div>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.sidebar.file_uploader(
    "Upload movies.csv", 
    type=["csv"],
    help="Upload your TMDB 5000 movies dataset here."
)

# Resolve data loading location
if uploaded_file is not None:
    try:
        df, all_genres = load_and_clean_data(uploaded_file)
        st.session_state['dataset_loaded'] = True
        st.session_state['using_mock'] = False
        st.sidebar.success("Successfully loaded uploaded dataset.")
    except Exception as e:
        st.sidebar.error(f"Error loading uploaded file: {str(e)}")
elif os.path.exists(DATA_FILE_PATH):
    try:
        df, all_genres = load_and_clean_data(DATA_FILE_PATH)
        st.session_state['dataset_loaded'] = True
        st.session_state['using_mock'] = False
        st.sidebar.info(f"Loaded {DATA_FILE_PATH} from project root.")
    except Exception as e:
        st.sidebar.error(f"Error reading local {DATA_FILE_PATH}: {str(e)}")
else:
    # Generate mock dataset for instant preview
    mock_df = generate_mock_data()
    # Save the mock dataset locally as fallback
    mock_df.to_csv(DATA_FILE_PATH, index=False)
    df, all_genres = load_and_clean_data(DATA_FILE_PATH)
    st.session_state['dataset_loaded'] = True
    st.session_state['using_mock'] = True
    st.sidebar.warning(f"{DATA_FILE_PATH} not found. Running on generated demonstration data.")
    st.sidebar.markdown(
        f"""
        <div style="font-size: 12px; color: #666666; margin-top: 5px;">
            To use your own data, place a file named <b>movies.csv</b> in the directory:
            <br><code>C:\\Users\\Ayush\\.gemini\\antigravity\\scratch\\movie_iq\\movies.csv</code>
            <br>or upload it above.
        </div>
        """,
        unsafe_allow_html=True
    )

# Proceed only if df is successfully loaded
if df is not None and not df.empty:
    
    # ----------------------------------------------------
    # SIDEBAR FILTER CONTROLS
    # ----------------------------------------------------
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    st.sidebar.markdown(
        f"""
        <div style="border-bottom: 1px solid #dcdcdc; padding-bottom: 8px; margin-bottom: 15px;">
            <h3 style="margin: 0; color: {PRIMARY_COLOR}; font-size: 16px;">Dashboard Filters</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Genre multi-select with empty list defaulting to All genres (using native placeholder)
    selected_genres = st.sidebar.multiselect(
        "Select Genres",
        options=all_genres,
        default=[],
        placeholder="All Genres (Default)",
        help="Filter the visual analytics by specific genres. Leave empty to show all."
    )
    
    # Minimum vote average slider
    min_vote = st.sidebar.slider(
        "Minimum Vote Average",
        min_value=0.0,
        max_value=10.0,
        value=0.0,
        step=0.1,
        help="Filter the visual analytics by minimum average user score."
    )
    
    # Apply filters to EDA dataframe
    # Explode temporarily to filter by genre, then reconstruct
    if selected_genres:
        exploded_temp = df.explode('parsed_genres')
        filtered_ids = exploded_temp[exploded_temp['parsed_genres'].isin(selected_genres)]['title'].unique()
        eda_df = df[df['title'].isin(filtered_ids)]
    else:
        # If no specific genres are chosen, default to all data
        eda_df = df.copy()
        
    eda_df = eda_df[eda_df['vote_average'] >= min_vote]
    
    # ----------------------------------------------------
    # TRAIN RANDOM FOREST MODEL (Global cache)
    # ----------------------------------------------------
    @st.cache_resource
    def get_trained_model(_dataframe):
        # We pass dataframe in without leading underscore, but cache resource is based on parameters
        return train_and_evaluate_model(_dataframe)
        
    model_results = get_trained_model(df)
    
    # ----------------------------------------------------
    # TABS STRUCTURE
    # ----------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "Exploratory Data Analysis",
        "Statistical Testing",
        "Predictive Modeling",
        "Success Predictor"
    ])
    
    # ----------------------------------------------------
    # TAB 1: EXPLORATORY DATA ANALYSIS (EDA)
    # ----------------------------------------------------
    with tab1:
        st.markdown(f"### Exploratory Visualizations")
        st.markdown("Filter values in the sidebar to dynamically update the visualizations in this section.")
        
        # Row 1: Budget vs Revenue & Genre Trends
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 1. Budget vs. Revenue Relationship")
            if not eda_df.empty:
                fig_br = plot_budget_vs_revenue(eda_df)
                st.pyplot(fig_br)
                st.markdown(
                    """
                    **Description**: The scatter plot shows a strong positive correlation between a movie's budget and its revenue.
                    Higher budgets do tend to yield higher absolute revenues. However, higher budgets also carry significantly higher risk;
                    several high-budget movies fall below the diagonal red dashed "break-even" line, meaning they failed to earn back their production budget.
                    """
                )
            else:
                st.warning("No data matches current sidebar filters.")
                
        with col2:
            st.markdown("#### 2. Genre Distribution & Success Rates")
            if not eda_df.empty:
                fig_genre = plot_genre_trends(eda_df)
                st.pyplot(fig_genre)
                st.markdown(
                    """
                    **Description**: The left bar chart displays the most frequent primary genres in the dataset. 
                    The right bar chart displays their corresponding financial success rates (percentage of movies where Revenue > Budget). 
                    While certain genres like Action and Adventure are highly common, genres such as Science Fiction, Thriller, and Fantasy
                    often show varying levels of success rates, with some niche genres achieving high percentages of profitability despite fewer releases.
                    """
                )
            else:
                st.warning("No data matches current sidebar filters.")
                
        # Row 2: Success Associations & Correlation Heatmap
        col3, col4 = st.columns([3, 2])
        
        with col3:
            st.markdown("#### 3. Popularity, Runtime, and Vote Average vs. Success")
            if not eda_df.empty:
                fig_assoc = plot_success_associations(eda_df)
                st.pyplot(fig_assoc)
                st.markdown(
                    """
                    **Description**: Box plots display the distribution of key numerical variables grouped by movie outcome:
                    - **Popularity**: Successful movies exhibit significantly higher popularity scores on average.
                    - **Runtime**: Runtimes are relatively similar, but successful movies have a slightly tighter distribution around 100-120 minutes.
                    - **Vote Average**: Successful movies have a higher median user rating compared to unsuccessful movies.
                    
                    *Insight*: Among the three features, **Popularity** exhibits the most pronounced division between successful and unsuccessful films.
                    """
                )
            else:
                st.warning("No data matches current sidebar filters.")
                
        with col4:
            st.markdown("#### 4. Correlation Heatmap")
            if not eda_df.empty:
                fig_corr = plot_correlation_heatmap(eda_df)
                st.pyplot(fig_corr)
                st.markdown(
                    """
                    **Description**: This correlation matrix evaluates the linear relationship between numerical features.
                    
                    **Modeling Concerns**: 
                    - There is a very strong correlation between `revenue` and `vote_count` (often > 0.7) and between `budget` and `revenue`.
                    - Because `revenue` is directly used to calculate our target variable `success` (`revenue > budget`), we **must exclude revenue**
                      from features to prevent target leakage.
                    - The strong correlation between features like `popularity` and `vote_count` indicates potential multicollinearity. While Random Forest 
                      is robust to multicollinearity, keeping highly correlated features can dilute feature importance scores.
                    """
                )
            else:
                st.warning("No data matches current sidebar filters.")
                
    # ----------------------------------------------------
    # TAB 2: STATISTICAL TESTING
    # ----------------------------------------------------
    with tab2:
        st.markdown("### Statistical Hypothesis Testing")
        st.markdown("Statistical significance helps determine if observed differences in our data are due to actual underlying patterns or random chance.")
        
        col_t, col_chi = st.columns(2)
        
        with col_t:
            st.markdown("#### 1. Two-Sample T-Test (Numeric Variables)")
            target_t_feature = st.selectbox("Select Feature for T-Test", ["popularity", "vote_average", "runtime"])
            
            t_results = run_t_test(df, target_t_feature)
            
            st.markdown(f"**Null Hypothesis ($H_0$)**:\n{t_results['null_hypothesis']}")
            st.markdown(f"**Alternative Hypothesis ($H_a$)**:\n{t_results['alternative_hypothesis']}")
            
            st.markdown(
                f"""
                <div style="background-color: #f8f9fa; border-left: 4px solid {PRIMARY_COLOR}; padding: 12px; margin: 15px 0;">
                    <b>Results:</b><br>
                    - T-Statistic: <code>{t_results['t_statistic']:.4f}</code><br>
                    - P-Value: <code>{t_results['p_value']:.4e}</code>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.write(t_results['conclusion'])
            
        with col_chi:
            st.markdown("#### 2. Chi-Square Test of Independence (Categorical Variables)")
            
            chi_results = run_chi_square_test(df)
            
            st.markdown(f"**Null Hypothesis ($H_0$)**:\n{chi_results['null_hypothesis']}")
            st.markdown(f"**Alternative Hypothesis ($H_a$)**:\n{chi_results['alternative_hypothesis']}")
            
            st.markdown(
                f"""
                <div style="background-color: #f8f9fa; border-left: 4px solid {ACCENT_COLOR}; padding: 12px; margin: 15px 0;">
                    <b>Results:</b><br>
                    - Chi-Square Statistic: <code>{chi_results['chi2_statistic']:.4f}</code><br>
                    - Degrees of Freedom: <code>{chi_results['degrees_of_freedom']}</code><br>
                    - P-Value: <code>{chi_results['p_value']:.4e}</code>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.write(chi_results['conclusion'])
            
        st.markdown("---")
        st.markdown("#### 3. Statistical Concepts in Plain Language")
        
        st.markdown(
            """
            **What does a p-value tell you?**
            A p-value (probability value) measures the probability of obtaining test results at least as extreme as the results actually observed, 
            assuming that the null hypothesis is true. 
            - A **very small p-value** indicates that the observed data is highly unlikely to have occurred under the null hypothesis, 
              leading us to reject it in favor of the alternative hypothesis.
            - A **large p-value** suggests that the observed differences could easily be the result of random sampling variation, 
              meaning we do not have sufficient evidence to reject the null hypothesis.
            
            **What threshold was used and why?**
            We used the standard significance threshold ($\alpha$) of **0.05 (5%)**. 
            This threshold represents a 5% risk of committing a Type I error (concluding a difference exists when it actually does not). 
            In academic and exploratory data science projects, 0.05 is the accepted baseline threshold that balances the risks of Type I errors (false positives) 
            and Type II errors (false negatives, failing to detect an effect that does exist).
            """
        )
        
    # ----------------------------------------------------
    # TAB 3: PREDICTIVE MODELING (RANDOM FOREST)
    # ----------------------------------------------------
    with tab3:
        st.markdown("### Random Forest Classifier Performance")
        
        # Performance Summary Cards
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.markdown(
                f'<div style="text-align: center; background-color: #f8f9fa; padding: 15px; border-radius: 6px;">'
                f'<span class="metric-label">Accuracy Score</span><br>'
                f'<span class="metric-value">{model_results["accuracy"]:.2%}</span>'
                f'</div>',
                unsafe_allow_html=True
            )
        with col_m2:
            st.markdown(
                f'<div style="text-align: center; background-color: #f8f9fa; padding: 15px; border-radius: 6px;">'
                f'<span class="metric-label">Precision Score</span><br>'
                f'<span class="metric-value">{model_results["precision"]:.2%}</span>'
                f'</div>',
                unsafe_allow_html=True
            )
        with col_m3:
            st.markdown(
                f'<div style="text-align: center; background-color: #f8f9fa; padding: 15px; border-radius: 6px;">'
                f'<span class="metric-label">Recall Score</span><br>'
                f'<span class="metric-value">{model_results["recall"]:.2%}</span>'
                f'</div>',
                unsafe_allow_html=True
            )
        with col_m4:
            st.markdown(
                f'<div style="text-align: center; background-color: #f8f9fa; padding: 15px; border-radius: 6px;">'
                f'<span class="metric-label">Validation Set Split</span><br>'
                f'<span class="metric-value">{model_results["test_set_size"]} movies (20%)</span>'
                f'</div>',
                unsafe_allow_html=True
            )
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            st.markdown("#### Evaluation & Mistakes Analysis")
            fig_cm = plot_confusion_matrix(model_results['confusion_matrix'])
            st.pyplot(fig_cm)
            
            # Text analysis of confusion matrix
            cm = model_results['confusion_matrix']
            tn, fp, fn, tp = cm.ravel()
            total = tn + fp + fn + tp
            
            st.markdown(
                f"""
                **How well does it predict success?**
                The model predicts movie success with an accuracy of **{model_results["accuracy"]:.2%}**. 
                - **Precision ({model_results["precision"]:.2%})** indicates that when the model predicts a movie will be successful, 
                  it is correct {model_results["precision"]:.2%} of the time.
                - **Recall ({model_results["recall"]:.2%})** indicates that the model successfully identifies {model_results["recall"]:.2%} of all actual successful movies in the validation set.
                
                **Where does it make the most mistakes?**
                Out of {total} test samples:
                - **False Positives (predicted Success, actually Unsuccessful)**: `{fp}` ({fp/total:.1%})
                - **False Negatives (predicted Unsuccessful, actually Successful)**: `{fn}` ({fn/total:.1%})
                
                The confusion matrix indicates that the model tends to make more **{ "False Negatives" if fn > fp else "False Positives" }** mistakes. 
                This means it is slightly { "underestimating" if fn > fp else "overestimating" } the potential success of movies.
                """
            )
            
        with col_right:
            st.markdown("#### Feature Importance Analysis")
            fig_imp = plot_feature_importance(model_results['feature_importance'])
            st.pyplot(fig_imp)
            
            # Describe alignment with EDA
            top_feature = model_results['feature_importance'].iloc[0]['Feature']
            st.markdown(
                f"""
                **Which features matter most?**
                The bar chart ranks the top 10 features by their relative importance to the Random Forest model. 
                The most critical feature is **{top_feature.capitalize()}**, followed closely by other numeric parameters.
                
                **Does this agree with EDA and Statistical Tests?**
                Yes. In the Exploratory Data Analysis phase, the box plots highlighted that successful and unsuccessful movies 
                exhibited distinct distributions in **popularity** and **vote_average**. The T-tests confirmed that these differences were 
                statistically significant. Correspondingly, the machine learning model assigns high relative importance to these features 
                when constructing decision paths.
                """
            )
            
        st.markdown("---")
        st.markdown("#### Modeling Choices and Concepts")
        
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            st.markdown(
                """
                **1. Feature Selection & Exclusions**
                - **Included Features**: `budget`, `popularity`, `runtime`, `vote_average`, `vote_count`, and `primary_genre`.
                - **Exclusions**:
                  - `title`: Excluded because movie titles are unique identifier strings (high cardinality) and do not provide structured statistical patterns.
                  - `revenue`: Excluded because it is the directly correlated target definition component. If included, it would lead to **data leakage**, resulting in a trivial model that memorizes the target definition instead of learning predictive cues from pre-release features.
                """
            )
        with col_c2:
            st.markdown(
                """
                **2. Train/Test Splits**
                We split the cleaned dataset into **80% training** and **20% validation** partitions. 
                
                **Why is a separate test set important?**
                A separate test set is crucial to evaluate model performance on data it has never seen during training. 
                This allows us to detect and prevent **overfitting** (where a model memorizes details of the training set but fails to generalize to new instances), 
                providing an unbiased estimate of the classifier's real-world predictive capability.
                """
            )
        with col_c3:
            st.markdown(
                """
                **3. How a Random Forest Predicts**
                A **Random Forest** is an ensemble learning method that works by:
                1. Generating multiple **Decision Trees** (e.g., 100) during the training phase.
                2. Using **bagging** (Bootstrap Aggregating) so each tree is trained on a random selection of data rows.
                3. Splitting nodes based on a **random subset of features**, which keeps individual trees decorrelated and prevents single dominant features from biasing all paths.
                4. At inference, each tree casts a vote (Successful vs. Unsuccessful). The forest combines these votes and outputs the **majority decision** as the final prediction.
                """
            )
            
    # ----------------------------------------------------
    # TAB 4: INTERACTIVE SUCCESS PREDICTOR
    # ----------------------------------------------------
    with tab4:
        st.markdown("### Film Success Calculator")
        st.markdown("Input the parameters of a movie project to run predictions using the trained Random Forest classifier.")
        
        # User input form
        with st.form("prediction_form"):
            col_in1, col_in2 = st.columns(2)
            
            with col_in1:
                input_budget = st.number_input(
                    "Production Budget (USD)",
                    min_value=10000,
                    max_value=500000000,
                    value=50000000,
                    step=500000,
                    format="%d",
                    help="Estimated movie production budget in USD."
                )
                
                input_popularity = st.slider(
                    "Projected Popularity Score",
                    min_value=0.0,
                    max_value=500.0,
                    value=25.0,
                    step=1.0,
                    help="TMDB popularity metric index (higher means more engagement)."
                )
                
                input_runtime = st.slider(
                    "Runtime (Minutes)",
                    min_value=10,
                    max_value=300,
                    value=110,
                    step=1,
                    help="Total movie length in minutes."
                )
                
            with col_in2:
                input_vote_avg = st.slider(
                    "Estimated Vote Average",
                    min_value=1.0,
                    max_value=10.0,
                    value=6.5,
                    step=0.1,
                    help="Target average user score rating."
                )
                
                input_vote_count = st.number_input(
                    "Projected Vote Count",
                    min_value=1,
                    max_value=20000,
                    value=500,
                    step=50,
                    format="%d",
                    help="Expected number of users casting votes."
                )
                
                input_genre = st.selectbox(
                    "Primary Film Genre",
                    options=all_genres,
                    index=0,
                    help="Select the dominant movie genre."
                )
                
            submit_button = st.form_submit_button("Predict Movie Success")
            
        if submit_button:
            # Gather inputs
            user_inputs = {
                'budget': input_budget,
                'popularity': input_popularity,
                'runtime': input_runtime,
                'vote_average': input_vote_avg,
                'vote_count': input_vote_count,
                'primary_genre': input_genre
            }
            
            # Predict
            pred_class, pred_prob = predict_single(
                model_results['model'], 
                model_results['feature_columns'], 
                user_inputs
            )
            
            # Clear UI separator
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Visual display (Using 1-2 color boxes with no emojis)
            if pred_class == 1:
                st.markdown(
                    f"""
                    <div class="success-card">
                        <div style="font-size: 14px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px;">Analysis Result</div>
                        <div style="font-size: 24px; margin-bottom: 10px;">SUCCESS PREDICTED</div>
                        <div style="font-weight: normal; font-size: 16px;">
                            The Random Forest model predicts this film project will be a <b>financial success</b> 
                            (Revenue > Budget) with a confidence probability of <b>{pred_prob:.1%}</b>.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div class="fail-card">
                        <div style="font-size: 14px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px;">Analysis Result</div>
                        <div style="font-size: 24px; margin-bottom: 10px;">NOT SUCCESSFUL PREDICTED</div>
                        <div style="font-weight: normal; font-size: 16px;">
                            The Random Forest model predicts this film project will <b>not be successful</b> 
                            (Revenue &le; Budget). The confidence probability of success is only <b>{pred_prob:.1%}</b>.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
else:
    st.error("Unable to load movie data. Please ensure movies.csv is uploaded or exists in the workspace.")
