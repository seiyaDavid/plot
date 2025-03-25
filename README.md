def create_and_show_plots(
    data: pd.DataFrame, stocks: List[str], all_results: Dict
) -> None:
    """Create professional plots using original CSV values and mark anomalies"""
    if all_results:
        n_stocks = len(stocks)
        n_cols = 3
        n_rows = (n_stocks + n_cols - 1) // n_cols

        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=[f"<b>{stock}</b>" for stock in stocks],
            horizontal_spacing=0.1,
            vertical_spacing=0.2,
        )

        for idx, stock in enumerate(stocks):
            predictions, scores = all_results[stock]
            row = idx // n_cols + 1
            col = idx % n_cols + 1

            # Plot original CSV values with better styling
            fig.add_trace(
                go.Scatter(
                    x=data["Date"],
                    y=data[stock],
                    mode="lines",
                    name=f"{stock} Values",
                    line=dict(color="#1E88E5", width=2),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

            # Add anomaly points with better styling
            anomaly_indices = np.where(predictions == -1)[0]
            if len(anomaly_indices) > 0:
                valid_indices = [i for i in anomaly_indices if i < len(data)]

                if valid_indices:
                    fig.add_trace(
                        go.Scatter(
                            x=data["Date"].iloc[valid_indices],
                            y=data[stock].iloc[valid_indices],
                            mode="markers",
                            name=f"{stock} Anomalies",
                            marker=dict(
                                color="#D32F2F",
                                size=10,
                                symbol="circle-open",
                                line=dict(width=2),
                            ),
                            showlegend=False,
                        ),
                        row=row,
                        col=col,
                    )

        # Update layout with better styling
        fig.update_layout(
            height=400 * n_rows,
            width=1400,
            title_text="<b>Stock Values and Detected Anomalies</b>",
            showlegend=False,
            title_x=0.5,
            margin=dict(l=50, r=50, t=100, b=50),
            paper_bgcolor="white",
            plot_bgcolor="#F5F5F5",
            font=dict(family="Arial, sans-serif", size=12),
        )

        # Update all axes for better appearance
        for i in range(1, n_rows * n_cols + 1):
            row_idx = (i - 1) // n_cols + 1
            col_idx = (i - 1) % n_cols + 1

            fig.update_xaxes(
                tickangle=45,
                gridcolor="#E0E0E0",
                zeroline=False,
                title_text="Date",
                row=row_idx,
                col=col_idx,
            )

            fig.update_yaxes(
                gridcolor="#E0E0E0",
                zeroline=False,
                title_text="Value",
                row=row_idx,
                col=col_idx,
            )

        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
===================================
"""
Stock Anomaly Detection Streamlit Application

This module provides a web interface for the stock anomaly detection system.
It allows users to:
    - Upload stock price data
    - Train/retrain anomaly detection models
    - Visualize anomalies in interactive plots
    - Download detected anomalies as CSV

The application uses:
    - Streamlit for the web interface
    - MLflow for model management
    - Plotly for interactive visualizations
"""

import streamlit as st
import pandas as pd
import mlflow
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from src.training.trainer import ModelTrainer
from src.utils.mlflow_utils import MLFlowManager
from src.data.data_loader import DataLoader
from src.utils.logger import setup_logger
import os
from typing import List, Dict

logger = setup_logger("streamlit_app")


def create_stock_plot(data, stock, predictions, anomaly_scores):
    """Create plotly figure for a stock"""
    fig = go.Figure()

    # Add original values
    fig.add_trace(
        go.Scatter(
            x=data["Date"],
            y=data[stock],
            mode="lines",
            name="Original Values",
            line=dict(color="blue"),
        )
    )

    # Add anomalies
    anomaly_indices = np.where(predictions == -1)[0]
    fig.add_trace(
        go.Scatter(
            x=data["Date"].iloc[anomaly_indices],
            y=data[stock].iloc[anomaly_indices],
            mode="markers",
            name="Anomalies",
            marker=dict(color="red", size=10),
        )
    )

    fig.update_layout(
        title=f"{stock} Values and Anomalies",
        xaxis_title="Date",
        yaxis_title="Value",
        showlegend=True,
    )

    return fig


def main():
    """
    Main application function that handles:
        - File upload interface
        - Model training controls
        - Progress tracking
        - Results visualization
        - Anomaly data export
    """
    try:
        # Set page config for a wider layout and professional title
        st.set_page_config(
            page_title="Stock Anomaly Detection System",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        # Add custom CSS for better styling
        st.markdown(
            """
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #1E88E5;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.5rem;
            color: #424242;
            margin-bottom: 1rem;
        }
        .info-text {
            font-size: 1rem;
            color: #616161;
        }
        .stProgress > div > div > div {
            background-color: #1E88E5;
        }
        .css-1v3fvcr {
            background-color: #f5f7f9;
        }
        .css-18e3th9 {
            padding-top: 2rem;
        }
        .stButton>button {
            background-color: #1E88E5;
            color: white;
            border-radius: 4px;
            padding: 0.5rem 1rem;
            font-weight: 500;
        }
        .stButton>button:hover {
            background-color: #1565C0;
        }
        .anomaly-highlight {
            color: #D32F2F;
            font-weight: bold;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

        # Create a header with logo and title
        col1, col2 = st.columns([1, 5])
        with col1:
            st.markdown("📈")
        with col2:
            st.markdown(
                '<div class="main-header">Stock Anomaly Detection System</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            '<div class="info-text">Upload stock data to detect anomalies using machine learning</div>',
            unsafe_allow_html=True,
        )

        # Create a sidebar for instructions and info
        with st.sidebar:
            st.markdown(
                '<div class="sub-header">Instructions</div>', unsafe_allow_html=True
            )
            st.markdown(
                """
            1. Upload a CSV file with stock data
            2. Choose a training option:
               - **Force Retrain**: Train new models for all stocks
               - **Retrain Missing**: Only train models for stocks without existing models
            3. View anomaly detection results and charts
            """
            )

            st.markdown('<div class="sub-header">About</div>', unsafe_allow_html=True)
            st.markdown(
                """
            This application uses Isolation Forest algorithm to detect anomalies in stock price data.
            
            Features:
            - Automated hyperparameter optimization
            - Model versioning with MLflow
            - Interactive visualization
            """
            )

        # Main content area with tabs
        tab1, tab2, tab3 = st.tabs(["Data Upload", "Model Training", "Results"])

        with tab1:
            st.markdown(
                '<div class="sub-header">Upload Stock Data</div>',
                unsafe_allow_html=True,
            )
            uploaded_file = st.file_uploader("Upload your stock data CSV", type=["csv"])

            if uploaded_file is not None:
                try:
                    data = pd.read_csv(uploaded_file)
                    stocks = [col for col in data.columns if col != "Date"]

                    # Show data preview
                    st.markdown(
                        '<div class="sub-header">Data Preview</div>',
                        unsafe_allow_html=True,
                    )
                    st.dataframe(data.head(10), use_container_width=True)

                    # Show data statistics
                    st.markdown(
                        '<div class="sub-header">Data Statistics</div>',
                        unsafe_allow_html=True,
                    )
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Number of Stocks", len(stocks))
                    with col2:
                        st.metric(
                            "Date Range",
                            f"{data['Date'].iloc[0]} to {data['Date'].iloc[-1]}",
                        )
                    with col3:
                        st.metric("Total Data Points", data.shape[0] * len(stocks))

                    # Create mlruns directory if it doesn't exist
                    os.makedirs("mlruns", exist_ok=True)

                    try:
                        logger.info("Initializing MLflow manager")
                        mlflow_manager = MLFlowManager("config/config.yaml")
                        logger.info("MLflow manager initialized successfully")

                        logger.info("Initializing model trainer")
                        trainer = ModelTrainer(
                            "config/config.yaml", "config/hyperparameters.yaml"
                        )
                        logger.info("Model trainer initialized successfully")

                        # Initialize results containers
                        all_results = {}
                        all_anomalies = []

                        # Create columns for status and progress
                        status_container = st.empty()
                        progress_bar = st.progress(0)

                        # Create two columns for buttons with better styling
                        col1, col2 = st.columns(2)

                        # Force retrain button in first column
                        train_all_btn = col1.button(
                            "Force Retrain All Models", key="train_all"
                        )

                        # Regular retrain button in second column
                        train_missing_btn = col2.button(
                            "Retrain Missing Models Only", key="train_missing"
                        )

                        # Handle button clicks
                        if train_all_btn:
                            with st.spinner("Force retraining all models..."):
                                st.warning(
                                    "Force retraining all models regardless of existing models..."
                                )
                                all_anomalies = []
                                all_results = {}

                                # Create a placeholder for the dynamic message
                                training_message = st.empty()

                                for idx, stock in enumerate(stocks):
                                    # Update message to show remaining stocks
                                    remaining_stocks = ", ".join(stocks[idx:])
                                    training_message.warning(
                                        f"Models remaining to train: {remaining_stocks}"
                                    )

                                    progress_bar.progress((idx + 1) / len(stocks))

                                    anomalies, predictions, scores = (
                                        trainer.train_stock_model(stock, data)
                                    )
                                    if not anomalies.empty:
                                        anomalies["is_anomaly"] = True
                                        all_anomalies.append(anomalies)
                                    all_results[stock] = (predictions, scores)

                                    # Update completion message
                                    if idx < len(stocks) - 1:
                                        training_message.success(
                                            f"Completed {stock}. Training remaining models..."
                                        )
                                    else:
                                        training_message.success(
                                            "All models trained successfully!"
                                        )
                                        status_container.empty()

                                if all_anomalies:
                                    final_anomalies = pd.concat(all_anomalies)
                                    final_anomalies.to_csv("anomalies.csv", index=False)
                                    st.success(
                                        "All models force retrained and anomalies saved!"
                                    )

                        elif train_missing_btn:
                            with st.spinner("Checking for missing models..."):
                                all_anomalies = []
                                all_results = {}
                                stocks_to_train = []

                                # Create a placeholder for the dynamic message
                                training_message = st.empty()

                                # Initialize DataLoader
                                data_loader = DataLoader(None)

                                # First, get existing models and their results
                                for stock in stocks:
                                    model = mlflow_manager.get_model(stock)
                                    if model is None:
                                        stocks_to_train.append(stock)
                                    else:
                                        # For existing models, get predictions and scores
                                        training_data, _, csv_values, _ = (
                                            data_loader.prepare_stock_data(data, stock)
                                        )
                                        predictions = model.predict(training_data)
                                        scores = model.score_samples(training_data)
                                        all_results[stock] = (predictions, scores)

                                        # Get anomalies for existing model
                                        anomaly_indices = np.where(predictions == -1)[0]
                                        if len(anomaly_indices) > 0:
                                            # Make sure we don't go out of bounds
                                            valid_indices = [
                                                i
                                                for i in anomaly_indices
                                                if i < len(data)
                                            ]

                                            if valid_indices:
                                                anomalies = pd.DataFrame(
                                                    {
                                                        "Date": data["Date"].iloc[
                                                            valid_indices
                                                        ],
                                                        f"{stock}_Value": csv_values.iloc[
                                                            valid_indices
                                                        ],
                                                        f"{stock}_PctChange": training_data.iloc[
                                                            valid_indices, 0
                                                        ].values,
                                                        f"{stock}_AnomalyScore": scores[
                                                            valid_indices
                                                        ],
                                                        f"{stock}_IsAnomaly": True,
                                                    }
                                                )
                                                all_anomalies.append(anomalies)

                                # Train missing models if any
                                if stocks_to_train:
                                    st.warning(
                                        f"Training models for: {', '.join(stocks_to_train)}"
                                    )

                                    for idx, stock in enumerate(stocks_to_train):
                                        # Update message to show remaining stocks
                                        remaining = ", ".join(stocks_to_train[idx:])
                                        training_message.warning(
                                            f"Models remaining to train: {remaining}"
                                        )

                                        progress_bar.progress(
                                            (idx + 1) / len(stocks_to_train)
                                        )

                                        # Train model and store results
                                        anomalies, predictions, scores = (
                                            trainer.train_stock_model(stock, data)
                                        )
                                        if not anomalies.empty:
                                            anomalies["is_anomaly"] = True
                                            all_anomalies.append(anomalies)
                                        all_results[stock] = (predictions, scores)

                                        # Update completion message
                                        if idx < len(stocks_to_train) - 1:
                                            training_message.success(
                                                f"Completed {stock}. Training remaining models..."
                                            )
                                        else:
                                            training_message.success(
                                                "All missing models trained successfully!"
                                            )
                                            status_container.empty()

                                # Create and show plots for all stocks
                                create_and_show_plots(data, stocks, all_results)

                                # Handle anomalies if any were found
                                if all_anomalies:
                                    final_anomalies = pd.concat(all_anomalies)
                                    final_anomalies.to_csv("anomalies.csv", index=False)

                                    if stocks_to_train:
                                        st.success(
                                            "New models trained and all anomalies saved!"
                                        )
                                    else:
                                        st.success(
                                            "Anomalies from existing models saved!"
                                        )

                                else:
                                    st.info("No anomalies detected in any models.")

                                if not stocks_to_train:
                                    st.info(
                                        "All models already exist. No retraining needed."
                                    )

                    except Exception as e:
                        logger.error(f"Error processing uploaded file: {str(e)}")
                        st.error(f"Error processing uploaded file: {str(e)}")
                except Exception as e:
                    logger.error(f"Error initializing MLflow or trainer: {str(e)}")
                    st.error(f"Error initializing MLflow or trainer: {str(e)}")
                    return

        with tab2:
            st.markdown(
                '<div class="sub-header">Model Training</div>', unsafe_allow_html=True
            )

            if "uploaded_file" not in locals() or uploaded_file is None:
                st.info("Please upload a CSV file in the Data Upload tab first.")
            else:
                st.success(
                    "Data loaded successfully. Use the buttons below to train models."
                )

                # Show model training instructions
                st.markdown(
                    """
                **Training Options:**
                - **Force Retrain All Models**: Retrain all models regardless of existing ones
                - **Retrain Missing Models Only**: Only train models for stocks without existing models
                """
                )

                # Show existing models if any
                try:
                    existing_models = []
                    for stock in stocks:
                        model = mlflow_manager.get_model(stock)
                        if model is not None:
                            existing_models.append(stock)

                    if existing_models:
                        st.markdown(
                            f"**Existing Models:** {', '.join(existing_models)}"
                        )
                    else:
                        st.warning(
                            "No existing models found. All stocks will be trained."
                        )
                except Exception as e:
                    st.error(f"Error checking existing models: {str(e)}")

        with tab3:
            st.markdown(
                '<div class="sub-header">Detection Results</div>',
                unsafe_allow_html=True,
            )

            if "all_results" in locals() and all_results:
                # Create metrics for anomaly summary
                anomaly_count = sum(
                    len(np.where(predictions == -1)[0])
                    for predictions, _ in all_results.values()
                )

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Stocks Analyzed", len(stocks))
                with col2:
                    st.metric(
                        "Stocks with Anomalies",
                        sum(
                            1
                            for predictions, _ in all_results.values()
                            if -1 in predictions
                        ),
                    )
                with col3:
                    st.metric("Total Anomalies Detected", anomaly_count)

                # Show anomalies dataframe if available
                if "final_anomalies" in locals() and not final_anomalies.empty:
                    st.markdown(
                        '<div class="sub-header">Anomaly Details</div>',
                        unsafe_allow_html=True,
                    )

                    # Add download button with better styling
                    col1, col2 = st.columns([1, 5])
                    with col1:
                        with open("anomalies.csv", "rb") as f:
                            st.download_button(
                                label="📥 Download CSV",
                                data=f,
                                file_name="anomalies.csv",
                                mime="text/csv",
                                help="Download the anomalies data as CSV",
                            )

                    # Show the dataframe
                    st.dataframe(final_anomalies, use_container_width=True)

                # Show plots
                st.markdown(
                    '<div class="sub-header">Visualization</div>',
                    unsafe_allow_html=True,
                )
                create_and_show_plots(data, stocks, all_results)
            else:
                st.info("Train models to see results here.")

    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error(f"Application error: {str(e)}")


def create_and_show_plots(
    data: pd.DataFrame, stocks: List[str], all_results: Dict
) -> None:
    """Create professional plots using original CSV values and mark anomalies"""
    if all_results:
        n_stocks = len(stocks)
        n_cols = 3
        n_rows = (n_stocks + n_cols - 1) // n_cols

        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=[f"<b>{stock}</b>" for stock in stocks],
            horizontal_spacing=0.1,
            vertical_spacing=0.2,
        )

        for idx, stock in enumerate(stocks):
            predictions, scores = all_results[stock]
            row = idx // n_cols + 1
            col = idx % n_cols + 1

            # Plot original CSV values with better styling
            fig.add_trace(
                go.Scatter(
                    x=data["Date"],
                    y=data[stock],
                    mode="lines",
                    name=f"{stock} Values",
                    line=dict(color="#1E88E5", width=2),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

            # Add anomaly points with better styling
            anomaly_indices = np.where(predictions == -1)[0]
            if len(anomaly_indices) > 0:
                valid_indices = [i for i in anomaly_indices if i < len(data)]

                if valid_indices:
                    fig.add_trace(
                        go.Scatter(
                            x=data["Date"].iloc[valid_indices],
                            y=data[stock].iloc[valid_indices],
                            mode="markers",
                            name=f"{stock} Anomalies",
                            marker=dict(
                                color="#D32F2F",
                                size=10,
                                symbol="circle-open",
                                line=dict(width=2),
                            ),
                            showlegend=False,
                        ),
                        row=row,
                        col=col,
                    )

        # Update layout with better styling
        fig.update_layout(
            height=400 * n_rows,
            width=1400,
            title_text="<b>Stock Values and Detected Anomalies</b>",
            showlegend=False,
            title_x=0.5,
            margin=dict(l=50, r=50, t=100, b=50),
            paper_bgcolor="white",
            plot_bgcolor="#F5F5F5",
            font=dict(family="Arial, sans-serif", size=12),
        )

        # Update all axes for better appearance
        for i in range(1, n_rows * n_cols + 1):
            row_idx = (i - 1) // n_cols + 1
            col_idx = (i - 1) % n_cols + 1

            fig.update_xaxes(
                tickangle=45,
                gridcolor="#E0E0E0",
                zeroline=False,
                title_text="Date",
                row=row_idx,
                col=col_idx,
            )

            fig.update_yaxes(
                gridcolor="#E0E0E0",
                zeroline=False,
                title_text="Value",
                row=row_idx,
                col=col_idx,
            )

        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()

