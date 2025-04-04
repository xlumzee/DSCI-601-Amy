import streamlit as st
import pandas as pd
import pickle
from data_loader import load_data  # Import the preprocessing function
from reinforce_agent import REINFORCEAgent
from main_reinforce import train_reinforce_agent, evaluate_reinforce_agent
import matplotlib.pyplot as plt
import config
import numpy as np


# Title
st.title("REINFORCE Trading Algorithm Dashboard")

# Sidebar: Dataset Upload
uploaded_file = st.sidebar.file_uploader("Upload Feature-Engineered Dataset (CSV)", type=["csv"])

st.sidebar.write("### Training Hyperparameters")
config.NUM_EPISODES = st.sidebar.number_input(
    "Number of Episodes", min_value=10, max_value=1000, value=config.NUM_EPISODES
)

# Learning rate giving errors
# config.AGENT_PARAMS = {
#     "learning_rate": st.sidebar.number_input("Learning Rate", min_value=0.0001, max_value=0.01, value=0.001, step=0.0001),
#     "gamma": st.sidebar.slider("Discount Factor (Gamma)", min_value=0.0, max_value=1.0, value=0.99),
# }

def max_drawdown(portfolio_values):
    """
    portfolio_values: list or array of cumulative portfolio values
    Returns the maximum drawdown as a fraction (e.g., 0.20 = 20%)
    """
    arr = np.array(portfolio_values)
    cumulative_max = np.maximum.accumulate(arr)
    drawdowns = (cumulative_max - arr) / cumulative_max
    return np.max(drawdowns)

if uploaded_file:
    try:
        
        data = load_data(uploaded_file)
        st.write("### Preprocessed Dataset Preview")
        st.dataframe(data.head())
    except ValueError as e:
        st.error(str(e))
        st.stop()

    # Training Hyperparameters
    #st.sidebar.write("### Training Hyperparameters")
    #episodes = st.sidebar.number_input("Number of Episodes", min_value=10, max_value=1000, value=50)
    # giving error - fix it
    #learning_rate = st.sidebar.number_input("Learning Rate", min_value=0.0001, max_value=0.01, value=0.001, step=0.0001)
    #gamma = st.sidebar.slider("Discount Factor (Gamma)", min_value=0.0, max_value=1.0, value=0.99)

    # Train Button
    if st.button("Start Training"):
        with st.spinner("Training in progress..."):
            train_rewards, trained_agent = train_reinforce_agent(data)
            st.success("Training Completed!")

            # Plot Training Rewards
            st.write("### Training Rewards per Episode")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(train_rewards)
            ax.set_xlabel("Episode")
            ax.set_ylabel("Total Reward")
            ax.set_title("Training Progress")
            st.pyplot(fig)

            # Save trained agent
            with open("trained_agent.pkl", "wb") as f:
                pickle.dump(trained_agent, f)

    # Evaluate Button - throwing error
    # if st.button("Evaluate Agent"):
    #     try:
    #         with open("trained_agent.pkl", "rb") as f:
    #             trained_agent = pickle.load(f)
    #         evaluation_results = evaluate_reinforce_agent(trained_agent, data)

    #         # Display Metrics and Results
    #         st.write(f"### Cumulative Return: {evaluation_results['cumulative_return']:.2f}%")
    #         st.write("### Portfolio Value Over Time")
    #         fig, ax = plt.subplots(figsize=(10, 5))
    #         ax.plot(evaluation_results['portfolio_values'], label="Portfolio Value")
    #         ax.set_xlabel("Time Steps")
    #         ax.set_ylabel("Portfolio Value")
    #         ax.legend()
    #         st.pyplot(fig)

    #     except FileNotFoundError:
    #         st.error("No trained agent found. Please train the agent first.")


# define the metrics - in a file called metrics.py that is being sourced from the dataset [data]
    if st.button("Evaluate Agent"):
        try:
            with open("trained_agent.pkl", "rb") as f:
                trained_agent = pickle.load(f)
            evaluation_results = evaluate_reinforce_agent(trained_agent, data)

            # Display your basic results (cumulative return, portfolio value chart)
            st.write(f"### Cumulative Return: {evaluation_results['cumulative_return']:.2f}%")

            st.write("### Portfolio Value Over Time")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(evaluation_results['portfolio_values'], label="Portfolio Value")
            ax.set_xlabel("Time Steps")
            ax.set_ylabel("Portfolio Value")
            ax.legend()
            st.pyplot(fig)

            # ----------------------------------
            # 1) Add a dropdown for metric choice
            # ----------------------------------
            metrics = [
                "Sharpe Ratio",
                "Maximum Drawdown",
                "Volatility (Std Dev of Returns)",
                "Distribution of Daily Returns",
                "Win Rate"
            ]
            selected_metric = st.selectbox("Select a metric to display:", metrics)

            # ----------------------------------
            # 2) Compute & display the chosen metric
            # ----------------------------------
            # Let's assume your `evaluation_results` might contain:
            # - evaluation_results['daily_returns'] 
            # - evaluation_results['portfolio_values']
            # - evaluation_results['trade_profits'] (list of P&L for each trade)
            # Adjust as needed for your actual returns/trades keys.

            if selected_metric == "Sharpe Ratio":
                if 'daily_returns' in evaluation_results:
                    daily_returns = evaluation_results['daily_returns']
                    avg_return = np.mean(daily_returns)
                    std_return = np.std(daily_returns)
                    risk_free_rate = 0.0
                    sharpe_ratio = (avg_return - risk_free_rate) / (std_return + 1e-9)
                    st.write(f"**Sharpe Ratio**: {sharpe_ratio:.2f}")
                else:
                    st.write("**daily_returns** not found in evaluation_results.")

            elif selected_metric == "Maximum Drawdown":
                if 'portfolio_values' in evaluation_results:
                    mdd = max_drawdown(evaluation_results['portfolio_values'])
                    st.write(f"**Maximum Drawdown**: {mdd*100:.2f}%")
                else:
                    st.write("**portfolio_values** not found in evaluation_results.")

            elif selected_metric == "Volatility (Std Dev of Returns)":
                if 'daily_returns' in evaluation_results:
                    vol = np.std(evaluation_results['daily_returns'])
                    st.write(f"**Volatility** (Std Dev): {vol:.4f}")
                else:
                    st.write("**daily_returns** not found in evaluation_results.")

            elif selected_metric == "Distribution of Daily Returns":
                if 'daily_returns' in evaluation_results:
                    daily_returns = evaluation_results['daily_returns']
                    fig2, ax2 = plt.subplots()
                    ax2.hist(daily_returns, bins=30)
                    ax2.set_title("Distribution of Daily Returns")
                    ax2.set_xlabel("Return")
                    ax2.set_ylabel("Frequency")
                    st.pyplot(fig2)
                else:
                    st.write("**daily_returns** not found in evaluation_results.")

            elif selected_metric == "Win Rate":
                if 'trade_profits' in evaluation_results:
                    trade_profits = evaluation_results['trade_profits']
                    num_wins = sum(1 for p in trade_profits if p > 0)
                    win_rate = (num_wins / len(trade_profits)) * 100
                    st.write(f"**Win Rate**: {win_rate:.2f}%")
                else:
                    st.write("**trade_profits** not found in evaluation_results.")

        except FileNotFoundError:
            st.error("No trained agent found. Please train the agent first.")
