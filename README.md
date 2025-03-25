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
