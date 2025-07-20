import yfinance as yf
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm 

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
        leg_prices = {}
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
            leg_prices[f"{leg.type.capitalize()} @ {leg.K}"] = leg_price
            total += leg.qty * leg.direction * leg_price
        return total, leg_prices 
    
                
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
            'vega': total_vega / 100, 
            'theta': total_theta / 365
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
    
    def calculate_max_profit_loss(self, net_premium):
        strikes = sorted([leg.K for leg in self.legs])
        test_prices = [0.01, 0.1, 1.0] + strikes + [10000, 100000]
        
        max_profit = float('-inf')
        max_loss = float('inf')
        
        for S in test_prices:
            pnl = self.payoff_at_expiry(S) - net_premium
            max_profit = max(max_profit, pnl)
            max_loss = min(max_loss, pnl)
        
        # For very high stock prices
        high_pnl = self.payoff_at_expiry(100000.0) - net_premium
        very_high_pnl = self.payoff_at_expiry(1000000.0) - net_premium

        if abs(very_high_pnl - high_pnl) > 0.001: 
            if very_high_pnl > high_pnl:
                max_profit = float('inf')
            elif very_high_pnl < high_pnl:
                max_loss = float('-inf')
    
        # For very low stock prices (approaching 0)
        low_pnl = self.payoff_at_expiry(0.01) - net_premium
        very_low_pnl = self.payoff_at_expiry(0.001) - net_premium
        
        if abs(very_low_pnl - low_pnl) > 0.001: 
            if very_low_pnl > low_pnl:
                max_profit = float('inf')
            elif very_low_pnl < low_pnl:
                max_loss = float('-inf')
        
        return max_profit, max_loss
    
def fetch_spot_price(ticker):
    try:
        data = yf.Ticker(ticker)
        return data.history(period="1d")['Close'].iloc[-1]
    except Exception as e:
        print(f"Error fetching price for {ticker}: {str(e)}")
        return None

def simulate_pnl_exipiration(strategy, S_range, net_premium):
    return [strategy.payoff_at_expiry(S) - net_premium for S in S_range]

def simulate_pnl_today(strategy, S_range, T, r, sigma, net_premium):
    pnl = []
    for S in S_range:
        try:
            total, _ = strategy.price(S, T, r, sigma)
            pnl.append(total - net_premium)
        except:
            pnl.append(0)
    return pnl

def plot_results(S_range, payoff, price, spot_price):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=S_range, y=payoff, mode='lines', name='P&L at Expiration'))
    fig.add_trace(go.Scatter(x=S_range, y=price, mode='lines', name='P&L Today'))
    fig.add_vline(x=spot_price, line_width=2, line_dash="dash", line_color="green", annotation_text=f"Spot: ${spot_price:.2f}")
    fig.add_hline(y=0,line_width=2, line_dash="dash", line_color="red")
    fig.update_layout(
        xaxis_title="Underlying Price",
        yaxis_title="Profit / Loss",
        height=550
    )

    return fig
    
def plot_3d_surface(strategy, S_range, T_range, r, sigma):
    S_mesh, T_mesh = np.meshgrid(S_range, T_range)
    Z = np.zeros_like(S_mesh)
    
    for i, T_val in enumerate(T_range):
        for j, S_val in enumerate(S_range):
            try:
                total, _ = strategy.price(S_val, T_val, r, sigma)
                Z[i, j] = total
            except:
                Z[i, j] = 0
    
    fig = go.Figure(data=[go.Surface(z=Z, x=S_mesh, y=T_mesh, colorscale='Viridis')])
    
    fig.update_layout(
        scene=dict(
            xaxis_title='Underlying Price',
            yaxis_title='Time to Expiry',
            zaxis_title='Strategy Value'
        ),
        height = 550
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
    for col in range(1, 5):
        fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="gray", row=1, col=col)
    fig.update_layout(
        height=400,
        showlegend=False,
        template='plotly_white'
    )

    fig.update_xaxes(title_text="Underlying Asset Price")

    return fig

# Template strategies
def create_template_strategy(strategy_name, spot_price):
    templates = {
        "Bull Call Spread": [
            OptionLeg("call", round(spot_price * 0.98, 2), 1, 1),  # Buy ITM call
            OptionLeg("call", round(spot_price * 1.05, 2), 1, -1)  # Sell OTM call
        ],
        "Bear Put Spread": [
            OptionLeg("put", round(spot_price * 1.02, 2), 1, 1),   # Buy ITM put
            OptionLeg("put", round(spot_price * 0.95, 2), 1, -1)   # Sell OTM put
        ],
        "Long Straddle": [
            OptionLeg("call", round(spot_price, 2), 1, 1),         # Buy ATM call
            OptionLeg("put", round(spot_price, 2), 1, 1)           # Buy ATM put
        ],
        "Short Straddle": [
            OptionLeg("call", round(spot_price, 2), 1, -1),        # Sell ATM call
            OptionLeg("put", round(spot_price, 2), 1, -1)          # Sell ATM put
        ],
        "Long Strangle": [
            OptionLeg("call", round(spot_price * 1.03, 2), 1, 1),  # Buy OTM call
            OptionLeg("put", round(spot_price * 0.97, 2), 1, 1)    # Buy OTM put
        ],
        "Short Strangle": [
            OptionLeg("call", round(spot_price * 1.03, 2), 1, -1), # Sell OTM call
            OptionLeg("put", round(spot_price * 0.97, 2), 1, -1)   # Sell OTM put
        ],
        "Iron Condor": [
            OptionLeg("put", round(spot_price * 0.92, 2), 1, 1),   # Buy OTM put
            OptionLeg("put", round(spot_price * 0.96, 2), 1, -1),  # Sell put
            OptionLeg("call", round(spot_price * 1.04, 2), 1, -1), # Sell call
            OptionLeg("call", round(spot_price * 1.08, 2), 1, 1)   # Buy OTM call
        ]
    }
    return templates.get(strategy_name, [])
