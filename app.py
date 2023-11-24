import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import plotly.express as px

# Initialize variables
independent_columns = None
dependent_column = None
X_train = None
X_test = None
y_train = None
y_test = None
selected_model = None
model = None  # Declare model here to make it accessible in the entire script

# Set page configuration
st.set_page_config(page_title="Group 2.1 Machine Learning App", layout="wide")

st.title("Group 2.1 Web-based Machine Learning App")

# Sidebar with custom styling
uploaded_file = st.file_uploader("Task 1: Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
else:
    st.warning("Please upload a CSV file.")

# Selecting Independent and Dependent Data
if uploaded_file is not None:
    col1, col2 = st.columns(2)

    with col1:
        independent_columns = st.multiselect(
            "Task 2: Select Independent Columns", df.columns)
    with col2:
        dependent_column = st.selectbox(
            "Task 2: Select Dependent Column", df.columns)

    st.subheader("Selected Columns:")
    st.write("Independent Columns:", independent_columns)
    st.write("Dependent Column:", dependent_column)

# Selecting Percentage Ratio of Training and Testing Data
if independent_columns and dependent_column:
    ratio_options = {"90:10": 0.1, "80:20": 0.2, "70:30": 0.3, "60:40": 0.4}
    selected_ratio = st.selectbox(
        "Task 3: Select Ratio of Training and Testing Data", list(ratio_options.keys()))

    st.subheader("Selected Ratio Information:")
    st.write(f"Selected Ratio: {selected_ratio}")

    # Split data into training and testing sets based on the selected ratio
    X_train, X_test, y_train, y_test = train_test_split(
        df[independent_columns],
        df[dependent_column],
        test_size=ratio_options[selected_ratio],
        random_state=42
    )

# Choosing a Machine Learning Model
if independent_columns and dependent_column:
    model_options = {"Random Forest": RandomForestRegressor(),
                     "XGBoost": XGBRegressor()}
    selected_model = st.selectbox(
        "Task 4: Select a Regression Model", list(model_options.keys()))

# Running the Model on Training and Testing Data and Evaluating Results
if independent_columns and dependent_column and selected_model:
    run_model_button = st.button("Task 5: Run Model")

    if run_model_button:
        # Train the selected model on the training data
        model = model_options[selected_model]
        model.fit(X_train, y_train)

        # Make predictions on the training and testing data
        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)

        # Evaluate the model
        train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))

        train_mae = mean_absolute_error(y_train, train_predictions)
        test_mae = mean_absolute_error(y_test, test_predictions)

        train_r2 = r2_score(y_train, train_predictions)
        test_r2 = r2_score(y_test, test_predictions)

        st.subheader("Model Evaluation Metrics:")
        st.write(f"Training RMSE: {train_rmse:.4f}")
        st.write(f"Testing RMSE: {test_rmse:.4f}")
        st.write(f"Training MAE: {train_mae:.4f}")
        st.write(f"Testing MAE: {test_mae:.4f}")
        st.write(f"Training R2: {train_r2:.4f}")
        st.write(f"Testing R2: {test_r2:.4f}")

# Displaying Feature Importance
if independent_columns and dependent_column and selected_model:
    calculate_feature_importance_button = st.button(
        "Task 7: Calculate Feature Importance")

    if calculate_feature_importance_button:
        model_for_feature_importance = model_options[selected_model]
        model_for_feature_importance.fit(
            df[independent_columns], df[dependent_column])

        if selected_model == "Random Forest":
            feature_importance = model_for_feature_importance.feature_importances_
        elif selected_model == "XGBoost":
            feature_importance = model_for_feature_importance.feature_importances_

        importance_df = pd.DataFrame({
            "Feature": independent_columns,
            "Importance": feature_importance
        })

        st.subheader("Feature Importance:")
        st.write(importance_df.sort_values(
            by="Importance", ascending=False))

# Calling New Data and Predicting
if independent_columns and dependent_column and selected_model:
    new_data_file = st.file_uploader(
        "Task 8: Upload new data (CSV format)", type=["csv"])

    if new_data_file is not None:
        new_data = pd.read_csv(new_data_file)

        if set(independent_columns) <= set(new_data.columns):
            st.subheader("Preview of New Data:")
            st.write(new_data.head())

            # Check if the model is fitted before making predictions
            if model:
                new_data_predictions = model.predict(
                    new_data[independent_columns])

                new_data["Prediction"] = new_data_predictions

                st.subheader("Predictions on New Data:")
                st.write(new_data)

                st.subheader("Density Heat Map:")
                fig = px.density_mapbox(
                    new_data,
                    lat='y',
                    lon='x',
                    z='Prediction',
                    radius=10,
                    center=dict(lat=np.mean(
                        new_data['y']), lon=np.mean(new_data['x'])),
                    zoom=10,
                    mapbox_style="stamen-terrain"
                )
                st.plotly_chart(fig)

# Main content area
st.subheader("Main Content Area")

# Additional content can be added here
