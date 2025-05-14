import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from io import StringIO
import time
import random

# Set page configuration
st.set_page_config(
    page_title="Adaptive AI Analytics Interface",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'expertise_level' not in st.session_state:
    st.session_state.expertise_level = "beginner"
if 'trust_level' not in st.session_state:
    st.session_state.trust_level = "medium"
if 'theme' not in st.session_state:
    st.session_state.theme = "light"
if 'large_text' not in st.session_state:
    st.session_state.large_text = False
if 'interactions' not in st.session_state:
    st.session_state.interactions = {
        'visualizations_viewed': 0,
        'advanced_features_used': 0,
        'help_accessed': 0
    }
if 'data' not in st.session_state:
    # Create a sample dataset if none is uploaded
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    sales = np.random.normal(loc=100, scale=20, size=100).cumsum() + 500
    expenses = sales * 0.6 + np.random.normal(loc=0, scale=50, size=100)
    profit = sales - expenses
    category = np.random.choice(['A', 'B', 'C', 'D'], size=100)
    region = np.random.choice(['North', 'South', 'East', 'West'], size=100)

    st.session_state.data = pd.DataFrame({
        'date': dates,
        'sales': sales,
        'expenses': expenses,
        'profit': profit,
        'category': category,
        'region': region
    })


# Apply custom CSS based on theme and accessibility preferences
def apply_custom_css():
    # Base theme styling
    if st.session_state.theme == "dark":
        background_color = "#121212"
        text_color = "#F0F0F0"
        accent_color = "#BB86FC"
    else:  # light theme
        background_color = "#FFFFFF"
        text_color = "#212121"
        accent_color = "#1E88E5"

    # Adjust text size based on accessibility preference
    text_size = "1.2rem" if st.session_state.large_text else "1rem"
    h1_size = "2.5rem" if st.session_state.large_text else "2rem"
    h2_size = "2rem" if st.session_state.large_text else "1.5rem"

    css = f"""
        <style>
            body {{
                color: {text_color};
                background-color: {background_color};
            }}
            .stApp {{
                background-color: {background_color};
            }}
            .stTextInput > div > div > input, 
            .stSelectbox, p, li {{
                font-size: {text_size} !important;
            }}
            h1, .stTitle {{
                font-size: {h1_size} !important;
            }}
            h2, h3 {{
                font-size: {h2_size} !important;
            }}
            .expertise-indicator {{
                padding: 5px 10px;
                border-radius: 5px;
                margin-bottom: 10px;
                animation: fadeIn 0.5s;
            }}
            @keyframes fadeIn {{
                0% {{ opacity: 0; }}
                100% {{ opacity: 1; }}
            }}
            .highlight {{
                border: 2px solid {accent_color};
                animation: pulse 2s infinite;
            }}
            @keyframes pulse {{
                0% {{ box-shadow: 0 0 0 0 rgba(204,169,44, 0.4); }}
                70% {{ box-shadow: 0 0 0 10px rgba(204,169,44, 0); }}
                100% {{ box-shadow: 0 0 0 0 rgba(204,169,44, 0); }}
            }}
            .tooltip {{
                position: relative;
                display: inline-block;
                border-bottom: 1px dotted #ccc;
            }}
            .tooltip .tooltiptext {{
                visibility: hidden;
                width: 200px;
                background-color: #555;
                color: #fff;
                text-align: center;
                border-radius: 6px;
                padding: 5px;
                position: absolute;
                z-index: 1;
                bottom: 125%;
                left: 50%;
                margin-left: -100px;
                opacity: 0;
                transition: opacity 0.3s;
            }}
            .tooltip:hover .tooltiptext {{
                visibility: visible;
                opacity: 1;
            }}
        </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# Apply custom CSS
apply_custom_css()


# Define adaptive help system
def show_help(section):
    st.session_state.interactions['help_accessed'] += 1
    update_expertise_based_on_interactions()

    if section == "upload":
        st.info("📎 Upload a CSV file to analyze. Your data should have columns for the values you want to visualize.")
    elif section == "visualization":
        if st.session_state.expertise_level == "beginner":
            st.info("""
            📊 **Visualization Help:**
            - Bar Chart: Shows comparison between categories
            - Line Chart: Shows trends over time
            - Scatter Plot: Shows relationship between two variables

            Try different combinations of columns to see how they look!
            """)
        else:
            st.info("""
            📊 **Visualization Options:**
            - Bar/Line charts work best for categorical or time-series data
            - Scatter plots reveal correlations between numeric variables
            - Consider using color encoding for additional dimensions
            - For time series data, try the decomposition analysis in the Advanced Features section
            """)
    elif section == "advanced":
        st.info("""
        🔬 **Advanced Analysis:**
        - Trend Analysis: Identifies patterns in time series data
        - Correlation Matrix: Shows relationships between all numeric variables
        - Predictive Model: Uses linear regression to predict values
        - PCA: Principal Component Analysis for dimensionality reduction

        These features can help you extract deeper insights from your data.
        """)


# Adaptive content based on expertise level
def get_chart_options_by_expertise():
    if st.session_state.expertise_level == "beginner":
        return ["Bar Chart", "Line Chart", "Scatter Plot"]
    elif st.session_state.expertise_level == "intermediate":
        return ["Bar Chart", "Line Chart", "Scatter Plot", "Histogram", "Box Plot"]
    else:  # expert
        return ["Bar Chart", "Line Chart", "Scatter Plot", "Histogram", "Box Plot",
                "Violin Plot", "Heatmap", "Pair Plot"]


# Update expertise level based on interactions
def update_expertise_based_on_interactions():
    interactions = st.session_state.interactions

    # Simple rule-based expertise assessment
    if interactions['advanced_features_used'] >= 5 and interactions['visualizations_viewed'] >= 10:
        st.session_state.expertise_level = "expert"
    elif interactions['advanced_features_used'] >= 2 or interactions['visualizations_viewed'] >= 5:
        st.session_state.expertise_level = "intermediate"
    else:
        st.session_state.expertise_level = "beginner"


# Function to increment interaction counters
def log_visualization_interaction():
    st.session_state.interactions['visualizations_viewed'] += 1
    update_expertise_based_on_interactions()


def log_advanced_feature_interaction():
    st.session_state.interactions['advanced_features_used'] += 1
    update_expertise_based_on_interactions()


# AI prediction with confidence scores
def ai_prediction(data, target_column, feature_columns, confidence_level):
    """
    Simulate AI prediction with confidence based on trust level
    """
    # Convert data to numeric
    X = data[feature_columns].apply(pd.to_numeric, errors='coerce')
    y = data[target_column].apply(pd.to_numeric, errors='coerce')

    # Drop rows with missing values
    valid_indices = X.dropna().index.intersection(y.dropna().index)
    X = X.loc[valid_indices]
    y = y.loc[valid_indices]

    # Perform basic checks on data
    if len(X) < 10:
        return {
            "error": "Not enough valid data points for prediction (minimum 10 required)",
            "recommendation": "Try another combination of columns or check your data for missing values"
        }

    # Split data into train/test
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Model performance
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Calculate confidence based on r2 and introduce some randomness
    # to simulate variability in model confidence
    base_confidence = max(0, min(1, (r2 + 1) / 2))  # Transform r2 to 0-1 range
    random_factor = np.random.normal(0, 0.05)  # Add some randomness
    model_confidence = max(0, min(1, base_confidence + random_factor))

    # Adjust confidence presentation based on trust level
    if confidence_level == "low":
        # Low trust means more conservative confidence estimate
        adjusted_confidence = model_confidence * 0.8
        confidence_range = (max(0, adjusted_confidence - 0.15),
                            min(1, adjusted_confidence + 0.15))
    elif confidence_level == "medium":
        # Medium trust gives moderate confidence adjustment
        adjusted_confidence = model_confidence * 0.9
        confidence_range = (max(0, adjusted_confidence - 0.1),
                            min(1, adjusted_confidence + 0.1))
    else:  # high trust
        # High trust gives confidence closer to the model's assessment
        adjusted_confidence = model_confidence
        confidence_range = (max(0, adjusted_confidence - 0.05),
                            min(1, adjusted_confidence + 0.05))

    return {
        "coefficients": dict(zip(feature_columns, model.coef_)),
        "intercept": model.intercept_,
        "mse": mse,
        "r2": r2,
        "confidence": adjusted_confidence,
        "confidence_range": confidence_range,
        "feature_importance": dict(zip(feature_columns, np.abs(model.coef_) / sum(np.abs(model.coef_)))),
    }


# Adaptive visualization with trust calibration
def create_adaptive_visualization(data, viz_type, x_col, y_col, color_col=None):
    st.session_state.interactions['visualizations_viewed'] += 1
    update_expertise_based_on_interactions()

    fig, ax = plt.subplots(figsize=(10, 6))

    try:
        if viz_type == "Bar Chart":
            sns.barplot(data=data, x=x_col, y=y_col, hue=color_col, ax=ax)
            plt.xticks(rotation=45)

        elif viz_type == "Line Chart":
            if color_col:
                for category, group in data.groupby(color_col):
                    ax.plot(group[x_col], group[y_col], label=category)
                ax.legend()
            else:
                ax.plot(data[x_col], data[y_col])
            plt.xticks(rotation=45)

        elif viz_type == "Scatter Plot":
            sns.scatterplot(data=data, x=x_col, y=y_col, hue=color_col, ax=ax)

        elif viz_type == "Histogram":
            sns.histplot(data=data, x=x_col, kde=True, ax=ax)

        elif viz_type == "Box Plot":
            sns.boxplot(data=data, x=color_col, y=y_col, ax=ax)
            plt.xticks(rotation=45)

        elif viz_type == "Violin Plot":
            sns.violinplot(data=data, x=color_col, y=y_col, ax=ax)
            plt.xticks(rotation=45)

        elif viz_type == "Heatmap":
            # For heatmap, we'll use correlation matrix of numeric columns
            numeric_data = data.select_dtypes(include=[np.number])
            sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', ax=ax)

        elif viz_type == "Pair Plot":
            # For pair plot, return a different kind of figure
            plt.close(fig)  # Close the previous figure
            if color_col:
                fig = sns.pairplot(data, hue=color_col, diag_kind='kde')
            else:
                fig = sns.pairplot(data, diag_kind='kde')

        # Add title and labels
        if viz_type != "Pair Plot" and viz_type != "Heatmap":
            ax.set_title(f"{viz_type} of {y_col} by {x_col}")
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)

        plt.tight_layout()
        return fig

    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        return None


# Function to display data insights based on expertise
def show_automated_insights(data):
    st.subheader("Automated Data Insights")

    with st.spinner("Analyzing data..."):
        # Basic insights for all levels
        st.write(f"📊 Dataset has {data.shape[0]} rows and {data.shape[1]} columns")

        # More detailed insights based on expertise
        if st.session_state.expertise_level in ["intermediate", "expert"]:
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                st.write("📈 Summary statistics for numeric columns:")
                st.dataframe(data[numeric_cols].describe())

                # Detect potential outliers
                outlier_info = {}
                for col in numeric_cols:
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    outlier_count = ((data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))).sum()
                    if outlier_count > 0:
                        outlier_info[col] = outlier_count

                if outlier_info:
                    st.write("⚠️ Potential outliers detected:")
                    for col, count in outlier_info.items():
                        st.write(f"  - {col}: {count} potential outlier(s)")

        if st.session_state.expertise_level == "expert":
            # Additional advanced insights for experts
            try:
                # Check for correlations
                numeric_data = data.select_dtypes(include=[np.number])
                if len(numeric_data.columns) > 1:
                    corr_matrix = numeric_data.corr()

                    # Find strong correlations (ignore self-correlations)
                    strong_corrs = []
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i + 1, len(corr_matrix.columns)):
                            if abs(corr_matrix.iloc[i, j]) > 0.7:
                                strong_corrs.append(
                                    (corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

                    if strong_corrs:
                        st.write("🔗 Strong correlations found:")
                        for col1, col2, corr in strong_corrs:
                            st.write(f"  - {col1} and {col2}: {corr:.2f}")

                # Check for missing values
                missing_vals = data.isnull().sum()
                if missing_vals.sum() > 0:
                    missing_cols = missing_vals[missing_vals > 0]
                    st.write("🕳️ Missing values detected:")
                    for col, count in missing_cols.items():
                        st.write(f"  - {col}: {count} missing value(s) ({count / len(data):.1%})")
            except Exception as e:
                st.warning(f"Could not complete advanced analysis: {str(e)}")


# Function to generate adaptive explanations of charts
def explain_chart(chart_type, x_col, y_col):
    if st.session_state.expertise_level == "beginner":
        if chart_type == "Bar Chart":
            return f"This bar chart shows the {y_col} for each {x_col}. Taller bars mean higher values."
        elif chart_type == "Line Chart":
            return f"This line chart shows how {y_col} changes over {x_col}. Look for upward or downward trends."
        elif chart_type == "Scatter Plot":
            return f"This scatter plot shows how {x_col} and {y_col} relate to each other. Clustered points suggest a relationship."
    else:
        if chart_type == "Bar Chart":
            return f"This bar chart visualizes the distribution of {y_col} across different {x_col} categories, allowing for easy comparison between groups."
        elif chart_type == "Line Chart":
            return f"This line chart depicts the temporal evolution of {y_col} with respect to {x_col}, highlighting trends, seasonality, and potential outliers."
        elif chart_type == "Scatter Plot":
            return f"This scatter plot reveals the relationship between {x_col} and {y_col}, where patterns may indicate correlation or underlying relationships."
        elif chart_type == "Histogram":
            return f"This histogram displays the distribution of {x_col}, showing frequency patterns and potential underlying distributions."
        elif chart_type == "Box Plot":
            return f"This box plot displays the statistical distribution of {y_col} across {x_col}, showing medians, quartiles, and outliers."
        elif chart_type == "Violin Plot":
            return f"This violin plot combines aspects of box plots and density plots to show the distribution of {y_col} across {x_col}."
        elif chart_type == "Heatmap":
            return "This heatmap visualizes the correlation matrix between numeric variables, with intense colors indicating stronger correlations."
        elif chart_type == "Pair Plot":
            return "This pair plot matrix shows relationships between multiple variables simultaneously, with scatter plots for pairwise relationships and distributions on the diagonal."

    return "This chart shows the relationship between your selected variables."


# Main application layout
def main():
    # Create sidebar for controls
    with st.sidebar:
        st.title("AI Analytics Dashboard")

        # User controls section
        st.subheader("User Controls")

        # Manual expertise level selector (for demo purposes)
        expertise_options = ["beginner", "intermediate", "expert"]
        expertise_level = st.selectbox(
            "Select your expertise level:",
            expertise_options,
            index=expertise_options.index(st.session_state.expertise_level)
        )

        if expertise_level != st.session_state.expertise_level:
            st.session_state.expertise_level = expertise_level
            st.experimental_rerun()

        # Trust level selector
        trust_options = ["low", "medium", "high"]
        trust_level = st.selectbox(
            "Select your AI trust level:",
            trust_options,
            index=trust_options.index(st.session_state.trust_level)
        )

        if trust_level != st.session_state.trust_level:
            st.session_state.trust_level = trust_level
            st.experimental_rerun()

        # Accessibility options
        st.subheader("Accessibility Options")

        # Theme selection
        theme = st.selectbox(
            "Select theme:",
            ["light", "dark"],
            index=0 if st.session_state.theme == "light" else 1
        )

        if theme != st.session_state.theme:
            st.session_state.theme = theme
            st.experimental_rerun()

        # Text size toggle
        large_text = st.checkbox("Large text mode", value=st.session_state.large_text)

        if large_text != st.session_state.large_text:
            st.session_state.large_text = large_text
            st.experimental_rerun()

        # Interaction metrics display
        if st.session_state.expertise_level == "expert":
            st.subheader("Interaction Metrics")
            st.write(f"Visualizations viewed: {st.session_state.interactions['visualizations_viewed']}")
            st.write(f"Advanced features used: {st.session_state.interactions['advanced_features_used']}")
            st.write(f"Help accessed: {st.session_state.interactions['help_accessed']}")

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        # Create a colorful expertise level indicator
        if st.session_state.expertise_level == "beginner":
            st.markdown("""
            <div class="expertise-indicator" style="background-color: #E8F4F8; color: #0077B6;">
                <strong>Beginner Mode:</strong> Simplified interface with helpful explanations
            </div>
            """, unsafe_allow_html=True)
        elif st.session_state.expertise_level == "intermediate":
            st.markdown("""
            <div class="expertise-indicator" style="background-color: #E8F8EB; color: #0A8754;">
                <strong>Intermediate Mode:</strong> Additional visualization options and analytics features
            </div>
            """, unsafe_allow_html=True)
        else:  # expert
            st.markdown("""
            <div class="expertise-indicator" style="background-color: #F8E8F8; color: #9A4EAE;">
                <strong>Expert Mode:</strong> Advanced analytics capabilities and detailed insights
            </div>
            """, unsafe_allow_html=True)

        st.title("Adaptive AI Analytics Dashboard")

        # Data upload section with tooltip explaining file formats
        st.header("Data Input")

        upload_col, help_col = st.columns([5, 1])
        with upload_col:
            st.markdown("""
            <div class="tooltip">Upload your CSV data file
              <span class="tooltiptext">Upload a CSV file with columns containing the data you want to analyze</span>
            </div>
            """, unsafe_allow_html=True)
            uploaded_file = st.file_uploader("", type="csv")

        with help_col:
            if st.button("Help 📘", key="upload_help"):
                show_help("upload")

        # Process uploaded file if available
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.session_state.data = data
                st.success(f"Data loaded successfully: {data.shape[0]} rows and {data.shape[1]} columns")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")

        # Display sample of the data
        st.subheader("Data Preview")
        st.dataframe(st.session_state.data.head())

        if st.session_state.data is not None:
            data = st.session_state.data

            # Show automated insights based on expertise level
            show_automated_insights(data)

    with col2:
        # Display interaction metrics for experts
        if st.session_state.expertise_level == "expert":
            st.subheader("Analysis Metrics")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Data Points", len(data))
            with col2:
                st.metric("Features", len(data.columns))

            # Show additional metrics for numeric columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                st.write("Numeric columns:")
                st.write(", ".join(numeric_cols))

        # Show trust level indicator
        st.subheader("AI Trust Calibration")

        trust_colors = {
            "low": "#FFF4E5",
            "medium": "#E8F4F8",
            "high": "#F0F8F0"
        }

        trust_text_colors = {
            "low": "#FF9800",
            "medium": "#0077B6",
            "high": "#4CAF50"
        }

        trust_descriptions = {
            "low": "The system will show detailed explanations and uncertainty ranges for AI predictions.",
            "medium": "The system will balance explanations with streamlined predictions.",
            "high": "The system will focus on efficient predictions with minimal explanations."
        }

        st.markdown(f"""
        <div style="background-color: {trust_colors[st.session_state.trust_level]}; 
                    color: {trust_text_colors[st.session_state.trust_level]}; 
                    padding: 10px; border-radius: 5px; margin-bottom: 10px;">
            <strong>Current Trust Level: {st.session_state.trust_level.capitalize()}</strong>
            <p style="font-size: 0.9em;">{trust_descriptions[st.session_state.trust_level]}</p>
        </div>
        """, unsafe_allow_html=True)

    # Visualization section
    st.header("Data Visualization")

    viz_col, help_col = st.columns([5, 1])
    with viz_col:
        st.write("Create visualizations based on your data")

    with help_col:
        if st.button("Help 📘", key="viz_help"):
            show_help("visualization")

    # Visualization controls
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        chart_options = get_chart_options_by_expertise()
        chart_type = st.selectbox("Chart Type", chart_options)

    with col2:
        x_column = st.selectbox("X-axis", st.session_state.data.columns)

    with col3:
        y_columns = st.session_state.data.select_dtypes(include=[np.number]).columns.tolist()
        if not y_columns:
            y_columns = st.session_state.data.columns.tolist()
        y_column = st.selectbox("Y-axis", y_columns)

    with col4:
        # Only show color option for some chart types
        if chart_type in ["Bar Chart", "Scatter Plot", "Box Plot", "Violin Plot", "Pair Plot"]:
            categorical_columns = ['None'] + st.session_state.data.select_dtypes(
                include=['object', 'category']).columns.tolist()
            color_column = st.selectbox("Color/Group By", categorical_columns)
            if color_column == 'None':
                color_column = None
        else:
            color_column = None

    # Generate the visualization
    if st.button("Generate Visualization"):
        with st.spinner("Creating visualization..."):
            fig = create_adaptive_visualization(
                st.session_state.data,
                chart_type,
                x_column,
                y_column,
                color_column
            )

            if fig is not None:
                st.pyplot(fig)

                # Show chart explanation based on expertise level
                explanation = explain_chart(chart_type, x_column, y_column)
                st.info(explanation)

                # Increment the visualization counter
                log_visualization_interaction()

    # Advanced features section - adaptive based on expertise level
    if st.session_state.expertise_level != "beginner":
        st.header("Advanced Analytics")

        adv_col, help_col = st.columns([5, 1])
        with adv_col:
            st.write("Explore deeper insights with advanced analytics tools")

        with help_col:
            if st.button("Help 📘", key="adv_help"):
                show_help("advanced")

        advanced_tabs = st.tabs(["Predictive Model", "Correlation Analysis", "Time Series Analysis", "PCA"])

        # 1. Predictive Model tab
        with advanced_tabs[0]:
            st.subheader("AI Predictive Modeling")
            st.write("Use AI to predict one variable based on others")

            pred_col1, pred_col2 = st.columns(2)

            with pred_col1:
                numeric_columns = st.session_state.data.select_dtypes(include=[np.number]).columns.tolist()
                target_col = st.selectbox("Target Variable (to predict)", numeric_columns)

            with pred_col2:
                available_features = [col for col in numeric_columns if col != target_col]
                feature_cols = st.multiselect("Feature Variables (predictors)", available_features,
                                              default=available_features[:min(2, len(available_features))])

            if st.button("Generate Predictive Model"):
                if not feature_cols:
                    st.warning("Please select at least one feature variable")
                else:
                    with st.spinner("Building predictive model..."):
                        # Log the advanced feature usage
                        log_advanced_feature_interaction()

                        # Generate prediction with confidence adjusted for trust level
                        result = ai_prediction(
                            st.session_state.data,
                            target_col,
                            feature_cols,
                            st.session_state.trust_level
                        )

                        if "error" in result:
                            st.error(result["error"])
                            st.info(result["recommendation"])
                        else:
                            # Display results based on trust level
                            st.subheader("Model Results")

                            # Format confidence for display
                            confidence_pct = f"{result['confidence'] * 100:.1f}%"

                            # Create columns for main results and confidence
                            res_col, conf_col = st.columns([3, 2])

                            with res_col:
                                st.write("**Model Performance:**")
                                st.write(f"R² Score: {result['r2']:.3f}")
                                st.write(f"Mean Squared Error: {result['mse']:.3f}")

                                if st.session_state.expertise_level == "expert":
                                    st.write("**Coefficients:**")
                                    for feature, coef in result["coefficients"].items():
                                        st.write(f"{feature}: {coef:.4f}")
                                    st.write(f"Intercept: {result['intercept']:.4f}")

                            with conf_col:
                                # Show confidence differently based on trust level
                                if st.session_state.trust_level == "low":
                                    st.write("**Model Confidence:**")
                                    st.write(f"Estimated accuracy: {confidence_pct}")
                                    st.write("**Confidence Range:**")
                                    st.write(
                                        f"{result['confidence_range'][0] * 100:.1f}% - {result['confidence_range'][1] * 100:.1f}%")
                                    st.write("*Note: Ranges indicate uncertainty in model predictions*")
                                elif st.session_state.trust_level == "medium":
                                    st.write("**Model Confidence:**")
                                    st.write(f"Estimated accuracy: {confidence_pct}")
                                    st.write(
                                        f"Range: ±{(result['confidence_range'][1] - result['confidence']) * 100:.1f}%")
                                else:  # high trust
                                    st.write("**Model Confidence:**")
                                    st.write(f"Estimated accuracy: {confidence_pct}")

                            # Feature importance visualization
                            st.subheader("Feature Importance")

                            # Create a bar chart of feature importance
                            importances = pd.DataFrame({
                                'Feature': list(result['feature_importance'].keys()),
                                'Importance': list(result['feature_importance'].values())
                            }).sort_values('Importance', ascending=False)

                            fig, ax = plt.subplots(figsize=(10, min(6, 1 + len(importances) * 0.3)))
                            sns.barplot(data=importances, y='Feature', x='Importance', ax=ax)
                            ax.set_title('Feature Importance')
                            st.pyplot(fig)

                            # Show interpretation based on trust level
                            if st.session_state.trust_level == "low":
                                st.info("""
                                **Interpretation Notes:**
                                - This model represents one possible relationship in your data
                                - The confidence score is an estimate and may not reflect actual prediction accuracy
                                - Consider these results as suggestive rather than definitive
                                - Multiple approaches should be used for robust conclusions
                                """)
                            elif st.session_state.trust_level == "medium":
                                st.info("""
                                **Interpretation Notes:**
                                - Feature importance indicates the relative influence of each variable
                                - Higher R² values indicate better fit (closer to 1.0)
                                - Consider model assumptions when interpreting results
                                """)

        # 2. Correlation Analysis tab
        with advanced_tabs[1]:
            st.subheader("Correlation Analysis")

            if st.button("Generate Correlation Matrix"):
                # Log the advanced feature usage
                log_advanced_feature_interaction()

                numeric_data = st.session_state.data.select_dtypes(include=[np.number])

                if len(numeric_data.columns) < 2:
                    st.warning("Need at least 2 numeric columns for correlation analysis")
                else:
                    corr_matrix = numeric_data.corr()

                    # Create correlation heatmap
                    fig, ax = plt.subplots(figsize=(10, 8))
                    heatmap = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
                    plt.title('Correlation Matrix')
                    st.pyplot(fig)

                    # Additional explanations based on trust level
                    if st.session_state.trust_level == "low":
                        st.info("""
                        **Understanding Correlations:**
                        - Values range from -1 (perfect negative correlation) to 1 (perfect positive correlation)
                        - 0 indicates no linear relationship
                        - Correlation does not imply causation
                        - These patterns may be influenced by outliers or non-linear relationships
                        """)
                    elif st.session_state.trust_level == "medium":
                        st.info("""
                        **Understanding Correlations:**
                        - Strong correlations (>0.7 or <-0.7) suggest significant relationships
                        - Consider investigating these relationships further with scatter plots
                        """)

                    # Find and display strongest correlations
                    if st.session_state.expertise_level == "expert":
                        # Get upper triangle of correlation matrix
                        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

                        # Find strongest absolute correlations
                        strongest_corrs = upper.unstack().sort_values(kind="quicksort", ascending=False)
                        strongest_corrs = strongest_corrs[strongest_corrs != 1.0]  # Remove self-correlations

                        if not strongest_corrs.empty:
                            st.subheader("Strongest Correlations")

                            # Display top 5 correlations or all if fewer than 5
                            top_n = min(5, len(strongest_corrs))
                            for i in range(top_n):
                                if i < len(strongest_corrs):
                                    pair = strongest_corrs.index[i]
                                    value = strongest_corrs.iloc[i]
                                    st.write(f"{pair[0]} — {pair[1]}: {value:.3f}")

        # 3. Time Series Analysis tab
        with advanced_tabs[2]:
            st.subheader("Time Series Analysis")

            # Automatically identify potential date columns
            date_cols = []
            for col in st.session_state.data.columns:
                # Check if column has date-like name
                if any(date_term in col.lower() for date_term in ['date', 'time', 'day', 'month', 'year']):
                    date_cols.append(col)

                # Try to convert to datetime (for already formatted dates)
                if col not in date_cols:
                    try:
                        pd.to_datetime(st.session_state.data[col].iloc[0])
                        date_cols.append(col)
                    except:
                        pass

            if not date_cols:
                st.warning("No date/time columns detected. Time series analysis requires a date column.")
            else:
                col1, col2 = st.columns(2)

                with col1:
                    date_column = st.selectbox("Date/Time Column", date_cols)

                with col2:
                    value_column = st.selectbox("Value to Analyze",
                                                st.session_state.data.select_dtypes(include=[np.number]).columns)

                if st.button("Analyze Time Series"):
                    # Log the advanced feature usage
                    log_advanced_feature_interaction()

                    with st.spinner("Analyzing time series data..."):
                        try:
                            # Create a copy of the data for processing
                            ts_data = st.session_state.data.copy()

                            # Ensure date column is in datetime format
                            try:
                                ts_data[date_column] = pd.to_datetime(ts_data[date_column])
                            except Exception as e:
                                st.error(f"Could not convert {date_column} to date format. Error: {str(e)}")
                                st.stop()

                            # Sort data by date
                            ts_data = ts_data.sort_values(by=date_column)

                            # Plot the time series
                            fig, ax = plt.subplots(figsize=(12, 6))
                            ax.plot(ts_data[date_column], ts_data[value_column])
                            ax.set_title(f'Time Series: {value_column} Over Time')
                            ax.set_xlabel(date_column)
                            ax.set_ylabel(value_column)
                            plt.xticks(rotation=45)
                            plt.tight_layout()
                            st.pyplot(fig)

                            # Calculate rolling statistics if we have enough data points
                            if len(ts_data) >= 10:
                                st.subheader("Rolling Statistics")

                                # Determine window size based on data length
                                window_size = max(3, len(ts_data) // 10)

                                # Calculate rolling mean and standard deviation
                                ts_data[f'Rolling Mean ({window_size})'] = ts_data[value_column].rolling(
                                    window=window_size).mean()
                                ts_data[f'Rolling Std ({window_size})'] = ts_data[value_column].rolling(
                                    window=window_size).std()

                                # Plot rolling statistics
                                fig, ax = plt.subplots(figsize=(12, 6))
                                ax.plot(ts_data[date_column], ts_data[value_column], label='Original')
                                ax.plot(ts_data[date_column], ts_data[f'Rolling Mean ({window_size})'],
                                        label=f'Rolling Mean ({window_size})')
                                ax.plot(ts_data[date_column], ts_data[f'Rolling Std ({window_size})'],
                                        label=f'Rolling Std ({window_size})')
                                ax.set_title(f'Rolling Statistics for {value_column}')
                                ax.set_xlabel(date_column)
                                ax.set_ylabel(value_column)
                                ax.legend()
                                plt.xticks(rotation=45)
                                plt.tight_layout()
                                st.pyplot(fig)

                                # If we have enough data points, show trend decomposition
                                if st.session_state.expertise_level == "expert" and len(ts_data) >= 30:
                                    try:
                                        from statsmodels.tsa.seasonal import seasonal_decompose

                                        # Set the date as index for decomposition
                                        ts_for_decompose = ts_data.set_index(date_column)[value_column]

                                        # Determine period for decomposition
                                        suggested_period = min(len(ts_for_decompose) // 4,
                                                               12)  # Default to quarterly or monthly
                                        period = st.slider("Select period for decomposition", 2,
                                                           min(len(ts_for_decompose) // 2, 30), suggested_period)

                                        # Perform decomposition
                                        decomposition = seasonal_decompose(ts_for_decompose, model='additive',
                                                                           period=period)

                                        # Plot decomposition
                                        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10))
                                        decomposition.observed.plot(ax=ax1)
                                        ax1.set_title('Observed')
                                        decomposition.trend.plot(ax=ax2)
                                        ax2.set_title('Trend')
                                        decomposition.seasonal.plot(ax=ax3)
                                        ax3.set_title('Seasonality')
                                        decomposition.resid.plot(ax=ax4)
                                        ax4.set_title('Residuals')
                                        plt.tight_layout()
                                        st.pyplot(fig)

                                        # Display explanations based on trust level
                                        if st.session_state.trust_level == "low":
                                            st.info("""
                                            **Understanding Time Series Decomposition:**
                                            - **Trend** shows the overall direction of the data over time
                                            - **Seasonality** shows repeating patterns at fixed intervals
                                            - **Residuals** show what remains after trend and seasonality are removed
                                            - This decomposition assumes additive components and may not capture all patterns
                                            - Different period values can significantly change results - try different values
                                            """)
                                        elif st.session_state.trust_level == "medium":
                                            st.info("""
                                            **Understanding Decomposition:**
                                            - Trend represents long-term progression
                                            - Seasonality shows cyclical patterns
                                            - Residuals should ideally look like random noise
                                            """)
                                    except Exception as e:
                                        st.error(f"Could not perform time series decomposition: {str(e)}")

                        except Exception as e:
                            st.error(f"Error in time series analysis: {str(e)}")

        # 4. PCA Analysis tab
        with advanced_tabs[3]:
            st.subheader("Principal Component Analysis (PCA)")
            st.write("Reduce dimensionality and identify patterns in your data")

            numeric_data = st.session_state.data.select_dtypes(include=[np.number])

            if len(numeric_data.columns) < 3:
                st.warning("PCA requires at least 3 numeric columns to be meaningful")
            else:
                # Select columns for PCA
                pca_columns = st.multiselect(
                    "Select numeric columns for PCA",
                    numeric_data.columns,
                    default=list(numeric_data.columns)[:min(5, len(numeric_data.columns))]
                )

                # Number of components
                n_components = st.slider(
                    "Number of components to extract",
                    min_value=2,
                    max_value=min(len(pca_columns), 10),
                    value=min(2, len(pca_columns))
                )

                if st.button("Run PCA Analysis"):
                    # Log the advanced feature usage
                    log_advanced_feature_interaction()

                    if len(pca_columns) < 2:
                        st.warning("Please select at least 2 columns for PCA")
                    else:
                        with st.spinner("Running PCA..."):
                            try:
                                # Prepare data for PCA
                                X = numeric_data[pca_columns].dropna()

                                if len(X) < 5:
                                    st.error("Not enough valid data points for PCA after removing missing values")
                                else:
                                    # Standardize the data
                                    scaler = StandardScaler()
                                    X_scaled = scaler.fit_transform(X)

                                    # Apply PCA
                                    pca = PCA(n_components=n_components)
                                    X_pca = pca.fit_transform(X_scaled)

                                    # Create DataFrame with PCA results
                                    pca_df = pd.DataFrame(
                                        data=X_pca,
                                        columns=[f'PC{i + 1}' for i in range(n_components)]
                                    )

                                    # Visualize PCA results
                                    st.subheader("PCA Results")

                                    # Show variance explained
                                    explained_variance = pca.explained_variance_ratio_
                                    total_variance = sum(explained_variance)

                                    st.write(f"Total variance explained: {total_variance:.2%}")

                                    # Plot explained variance
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    ax.bar(range(1, n_components + 1), explained_variance)
                                    ax.set_xlabel('Principal Component')
                                    ax.set_ylabel('Explained Variance Ratio')
                                    ax.set_title('Variance Explained by Each Principal Component')
                                    ax.set_xticks(range(1, n_components + 1))
                                    st.pyplot(fig)

                                    # Scatter plot of first two components
                                    fig, ax = plt.subplots(figsize=(10, 8))
                                    ax.scatter(pca_df['PC1'], pca_df['PC2'])
                                    ax.set_xlabel(f'PC1 ({explained_variance[0]:.2%} variance)')
                                    ax.set_ylabel(f'PC2 ({explained_variance[1]:.2%} variance)')
                                    ax.set_title('First Two Principal Components')

                                    # Add a grid
                                    ax.grid(True, linestyle='--', alpha=0.7)

                                    # Add the origin lines
                                    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
                                    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)

                                    st.pyplot(fig)

                                    # Show loadings (feature contributions)
                                    st.subheader("Feature Contributions")

                                    loadings = pca.components_
                                    loadings_df = pd.DataFrame(
                                        loadings.T,
                                        columns=[f'PC{i + 1}' for i in range(n_components)],
                                        index=pca_columns
                                    )

                                    st.dataframe(loadings_df)

                                    # Visualize feature contributions for first two components
                                    fig, ax = plt.subplots(figsize=(12, 8))

                                    # Draw arrows for feature loadings
                                    for i, feature in enumerate(pca_columns):
                                        ax.arrow(0, 0, loadings[0, i], loadings[1, i],
                                                 head_width=0.05, head_length=0.05, fc='blue', ec='blue')
                                        ax.text(loadings[0, i] * 1.15, loadings[1, i] * 1.15, feature,
                                                color='green', ha='center', va='center')

                                    # Add circle
                                    circle = plt.Circle((0, 0), 1, fill=False, color='red', linestyle='--')
                                    ax.add_patch(circle)

                                    # Set plot limits and labels
                                    plt.xlim(-1.1, 1.1)
                                    plt.ylim(-1.1, 1.1)
                                    plt.xlabel("PC1")
                                    plt.ylabel("PC2")
                                    plt.grid(True)
                                    plt.title("PCA Feature Contributions")

                                    # Add the origin lines
                                    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
                                    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)

                                    st.pyplot(fig)

                                    # Display explanations based on trust level
                                    if st.session_state.trust_level == "low":
                                        st.info("""
                                        **Understanding PCA Results:**
                                        - Principal components are new variables that capture variance in the data
                                        - The percentage values show how much information each component preserves
                                        - Feature arrows show how original variables contribute to components
                                        - Longer arrows indicate stronger influence
                                        - PCA assumes linear relationships and is sensitive to outliers
                                        - Results should be interpreted carefully alongside domain knowledge
                                        """)
                                    elif st.session_state.trust_level == "medium":
                                        st.info("""
                                        **Understanding PCA:**
                                        - Features pointing in similar directions are positively correlated
                                        - Features pointing in opposite directions are negatively correlated
                                        - Features at right angles have little correlation
                                        - Clusters in the scatter plot may indicate distinct groups in your data
                                        """)

                            except Exception as e:
                                st.error(f"Error in PCA analysis: {str(e)}")


# Run the application
if __name__ == "__main__":
    main()