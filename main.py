import yfinance as yf
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm 
import pandas as pd

class OptionLeg:
    def __init__(self, type_, K, qty, direction, expiry = None):
        self.type = type_  
        self.K = K
        self.qty = qty
        self.direction = direction  # +1 buy, -1 sell
        self.expiry = expiry

class Strategy:
    def __init__(self, legs):
        self.legs = legs

    def bs_d1_d2(self, S, K, T, r, sigma):
        d1 = (np.log(S/K) + (r+sigma**2/2)*T)/(sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2
    
    def price(self, S, T, r, sigma):
        total = 0
        for leg in self.legs:
            d1, d2, = self.bs_d1_d2(S, leg.K, T, r, sigma)
            K = leg.K
            try:
                if leg.type == 'call':
                    leg_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
                else: 
                    leg_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            except:
                leg_price = 0
            total += leg.qty * leg.direction * leg_price
        return total 

    def greeks(self, S, T, r, sigma):
        total_delta = total_gamma = total_vega = total_theta =  0

        for leg in self.legs:
            d1, d2 = self.bs_d1_d2(S, leg.K, T, r, sigma)
            K = leg.K

            if leg.type == 'call':
                delta = norm.cdf(d1)
                theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))) - r * K * np.exp(-r * T) * norm.cdf(d2)
            else:  
                delta = norm.cdf(d1) - 1
                theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))) + r * K * np.exp(-r * T) * norm.cdf(-d2)

            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            vega = S * norm.pdf(d1) * np.sqrt(T)
    
            total_delta += leg.qty * leg.direction * delta
            total_gamma += leg.qty * leg.direction * gamma
            total_theta += leg.qty * leg.direction * theta
            total_vega += leg.qty * leg.direction * vega
            
        return {
            'delta': total_delta,
            'gamma': total_gamma,
            'theta': total_theta / 365,  
            'vega': total_vega / 100,  
        }
    

    def payoff_at_expiry(self, S):
        total = 0
        for leg in self.legs:
            if leg.type == 'call':
                payoff = max(S - leg.K, 0)
            else:
                payoff = max(leg.K - S, 0)
            total += leg.qty * leg.direction * payoff
        return total
    
def fetch_spot_price(ticker):
    try:
        data = yf.Ticker(ticker)
        return data.history(period="1d")['Close'].iloc[-1]
    except Exception as e:
        print(f"Error fetching price for {ticker}: {str(e)}")
        return None

def simulate_pnl_exipiration(strategy, S_range, initial_cost):
    return [strategy.payoff_at_expiry(S) - initial_cost for S in S_range]

def simulate_pnl(strategy, S_range, T, r, sigma, initial_cost):
    return [strategy.price(S, T, r, sigma) - initial_cost for S in S_range]

def plot_results(S_range, payoff, price, spot_price):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=S_range, y=payoff, mode='lines', name='P&L at Expiration'))
    fig.add_trace(go.Scatter(x=S_range, y=price, mode='lines', name='P&L Today'))
    fig.add_vline(x=spot_price, line_width=2, line_dash="dash", line_color="green", name="Spot Price")
    fig.add_hline(y=0,line_width=2, line_dash="dash", line_color="red")
    fig.update_layout(
        xaxis_title="Underlying Price",
        yaxis_title="Profit / Loss",
        height=700
    )

    return fig
    
def plot_3d_surface(strategy, S_range, T_range, r, sigma):
    S_mesh, T_mesh = np.meshgrid(S_range, T_range)
    Z = np.zeros_like(S_mesh)
    
    for i, T_val in enumerate(T_range):
        for j, S_val in enumerate(S_range):
            try:
                Z[i, j] = strategy.price(S_val, T_val, r, sigma)
            except:
                Z[i, j] = 0
    
    fig = go.Figure(data=[go.Surface(z=Z, x=S_mesh, y=T_mesh, colorscale='Viridis')])
    
    fig.update_layout(
        scene=dict(
            xaxis_title='Underlying Price',
            yaxis_title='Time to Expiry',
            zaxis_title='Strategy Value'
        ),
        height = 700
    )
    
    return fig

def plot_greeks(strategy, S_range, T, r, sigma):
    deltas, gammas, vegas, thetas = [], [], [], []

    for s_price in S_range:
        greeks = strategy.greeks(s_price, T, r, sigma)
        deltas.append(greeks['delta'])
        gammas.append(greeks['gamma'])
        vegas.append(greeks['vega'])
        thetas.append(greeks['theta'])

    fig = make_subplots(
        rows=1, cols=4,
        subplot_titles=('Delta', 'Gamma', 'Vega', 'Theta'),
        shared_xaxes= True,
    )

    fig.add_trace(go.Scatter(x=S_range, y=deltas, mode='lines', name='Delta', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=S_range, y=gammas, mode='lines', name='Gamma', line=dict(color='green')), row=1, col=2)
    fig.add_trace(go.Scatter(x=S_range, y=vegas, mode='lines', name='Vega', line=dict(color='red')), row=1, col=3)
    fig.add_trace(go.Scatter(x=S_range, y=thetas, mode='lines', name='Theta', line=dict(color='purple')), row=1, col=4)

    fig.update_layout(
        height=600,
        showlegend=False,
        template='plotly_white'
    )

    fig.update_xaxes(title_text="Underlying Asset Price ($)")

    return fig

