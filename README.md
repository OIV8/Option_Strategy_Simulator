# Options Strategy Simulator & Greeks Visualizer

A Python-based web application for simulating multi-leg options strategies, visualizing payoffs, and analyzing key risk sensitivities (Greeks) across varying market conditions. Built with `Streamlit` and using a self-implemented Black-Scholes model for pricing and Greeks calculation.

## Features

- Build multi-leg options strategies (calls and puts, long or short)
- Visualize:
  - Payoff at expiration
  - P&L at current time (based on implied volatility)
  - Sensitivity plots: Delta, Gamma, Vega, Theta, Rho across spot prices
- Fetch real-time underlying asset prices from Yahoo Finance
- Interactive and intuitive dashboard using Streamlit

## Demo

![App Screenshot](screenshot.png)  <!-- Replace with your own screenshot path -->

## Installation

### Prerequisites
- Python 3.8+
- pip

### Clone and Setup
```bash
git clone https://github.com/YOUR_USERNAME/Options-Strategy-Simulator.git
cd Options-Strategy-Simulator
pip install -r requirements.txt
