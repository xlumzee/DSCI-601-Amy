# Optimizing Trading Strategies with Reinforcement Learning

Capstone DSCI-601 / 602  |  2024-2025

A reproducible research & engineering project that explores classical factor models and deep-reinforcement-learning (DRL) to design, train and back-test discretionary-free equity trading strategies.

<img width="791" height="522" alt="image" src="https://github.com/user-attachments/assets/d293f895-0c08-459f-96d9-bc956e0b415d" />


⸻

## Table of Contents
	1.	Project Overview
	2.	Quick Start
	3.	Repository Layout
	4.	Data Pipeline
	5.	Baseline Models
	6.	RL Environment & Agents
	7.	Back-testing & Evaluation
	8.	Unit Tests
	9.	Road-map
	10.	Contributing
	11.	License

⸻

## Project Overview

This capstone investigates whether deep-RL agents can reliably outperform conventional factor-based strategies on U.S. equities when realistic frictions (bid–ask spread, turnover, market impact) are included.

The workflow is divided into three layers:

Layer	Goal	Key Artefacts
Exploratory & Feature Engineering	Clean market micro-structure data, engineer lagged / rolling factors	Data/, DataPipelines/ notebooks & scripts
Classical Baseline	Benchmark with Random Forest, SVR, ARIMA forecasts	ProjectCode/baselines/
Reinforcement Learning	Train DQN / PPO agents inside a custom Gym environment that emits (state = engineered features) and rewards (risk-adjusted PnL)	ProjectCode/rl_env/, ProjectCode/train.py, ProjectCode/evaluate.py


⸻

### Quick Start

Tested on Python 3.12 (see  .python-version) and macOS/Ubuntu.

#### 1. clone
$ git clone https://github.com/xlumzee/Optimizing-Trading-Strategies-RL.git
$ cd Optimizing-Trading-Strategies-RL

#### 2. create & activate venv (recommended)
$ python3 -m venv .venv
$ source .venv/bin/activate

#### 3. install requirements
$ pip install -r requirements.txt  # ~220 MB incl. Jupyter & scikit-learn

#### 4. (optional) install torch with CUDA
$ pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

#### 5. spin up Jupyter to reproduce notebooks
$ jupyter lab

####6. kick-off an RL training run (defaults to DQN on S&P 500 mini-universe)
$ python ProjectCode/train.py --config configs/dqn_sp500.yaml

#### 7. back-test the saved checkpoint
$ python ProjectCode/evaluate.py --checkpoint runs/2025-07-01-dqn/best.pt --plot

The first training run (~15k environment steps) completes in under 10 minutes on an M1-Pro laptop; full 1-million-step runs take ~2 hours on a single RTX-4090.

⸻

Repository Layout

├── Data/                  # raw & processed CSVs (≈ 12 MB compressed)
│   ├── equities_raw/      # one CSV per ticker
│   └── features.parquet   # merged feature matrix after pipeline
│
├── DataPipelines/         # notebooks & py scripts for EDA + feature eng.
│   ├── DataViz_v2.ipynb
│   └── DataPrep_v2.ipynb
│
├── ProjectCode/
│   ├── rl_env/            # OpenAI Gym-compatible environment
│   ├── agents/            # DQN, PPO, A2C implementations (PyTorch)
│   ├── baselines/         # Classical ML benchmarks
│   ├── train.py           # CLI entry-point for RL training
│   └── evaluate.py        # generates back-test charts & metrics
│
├── tests/                 # lightweight unit tests for pipelines & envs
├── Research Papers/       # key literature (PDFs) that informed design
├── presentation/          # slides for academic defence
├── requirements.txt       # pinned, reproducible environment
└── README.md              # you are here


⸻

### Data Pipeline
	•	Source – Daily US equity data (2010-2024) pulled from Refinitiv, augmented with SP500 & DJI index returns.
	•	Feature Engineering – Lag-1–Lag-5 returns, rolling mean/vol, bid-ask spread, Amihud illiquidity, turnover and macro factors.
	•	Scripts – DataPrep_v2.ipynb transforms raw CSVs into a single features.parquet used by both classical and RL pipelines.

Tip — headless usage
Convert the notebook to a script and execute via:
jupyter nbconvert --to script DataPipelines/DataPrep_v2.ipynb && python DataPipelines/DataPrep_v2.py --save-parquet

⸻

### Baseline Models

Located in ProjectCode/baselines/:
	•	random_forest.py – feature importance sanity-check.
	•	svr.py – nonlinear benchmark.
	•	arima.py – time-series baseline.

Use python ProjectCode/baselines/run_all.py to reproduce the metrics table shown in the accompanying report.

⸻

### RL Environment & Agents
	•	Environment – rl_env/market_env.py inherits from gym.Env.
	•	State: engineered feature vector for the chosen ticker/universe.
	•	Actions: {-1 = short, 0 = flat, +1 = long}.
	•	Reward: daily log-return minus transaction costs.
	•	Agents – Tabular DQN, Dueling DQN, PPO and Actor-Critic variants in agents/.
	•	Config-driven – YAML files under configs/ let you swap networks, replay buffer sizes, learning rates, etc. without code edits.

⸻

### Back-testing & Evaluation

After training, run:

python ProjectCode/evaluate.py --checkpoint <path> --start 2023-01-01 --end 2024-12-31

Outputs
	1.	Equity-curve plot (cumulative returns)
	2.	Rolling Sharpe (252-day window)
	3.	Trade-by-trade summary (results/trades.csv)

Metrics are stored in a lightweight MLflow run so you can compare experiments visually. Use mlflow ui to launch the dashboard.


⸻

Contributing

Pull requests are welcome! Please open an issue first to discuss major changes.  Make sure new code passes flake8 and existing unit tests, and include docs / examples where appropriate.

⸻

License

This project is licensed under the MIT License – see LICENSE for details.

