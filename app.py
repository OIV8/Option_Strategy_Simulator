import streamlit as st
import main
import numpy as np

st.set_page_config(layout="wide")
st.title("Options Strategy Simulator & Greeks Visualizer")

#######################
#### Sidebar Inputs ###
#######################
if "legs" not in st.session_state:
    st.session_state.legs = []

@st.cache_data
def load_fetch_data(ticker):
    return main.fetch_spot_price(ticker)

ticker = st.sidebar.text_input("Enter Ticker", "AAPL")
spot = load_fetch_data(ticker)
st.sidebar.write(f"Current Spot Price: ${spot:.2f}")

T_days = st.sidebar.slider("Days to Expiry", 5, 90, 30)
T = T_days / 365
r = st.sidebar.slider("Risk-Free Rate (%)", 0.0, 5.0, 2.5) / 100
sigma = st.sidebar.slider("Volatility (%)", 5.0, 100.0, 25.0) / 100

# Template strategies
st.sidebar.markdown("---")
st.sidebar.write("Template strategies")
template_strategies = [
    "Bull Call Spread", "Bear Put Spread", "Long Straddle", "Short Straddle",
    "Long Strangle", "Short Strangle", "Iron Condor",
]
selected_template = st.sidebar.selectbox("Select a template strategy:", ["None"] + template_strategies)
if st.sidebar.button("Load Template", disabled=(selected_template == "None")):
        if selected_template != "None":
            st.session_state.legs = main.create_template_strategy(selected_template, spot)
            st.rerun()

# Custom strategies
st.sidebar.markdown("---")
st.sidebar.write("Add Custom Leg")
with st.sidebar.form("Add Option Leg"):
    leg_type = st.selectbox("Option Type", ["Call", "Put"], key="type")
    direction = st.selectbox("Direction", ["Buy", "Sell"], key="dir")
    K = st.number_input("Strike Price", value=round(spot), key="K_input")
    qty = st.number_input("Quantity", value=1, min_value=1, step=1, key="qty_input")
    add_leg = st.form_submit_button("Add Leg")

    if add_leg:
        dir_flag = 1 if direction == "Buy" else -1
        new_leg = main.OptionLeg(leg_type.lower(), K, qty, dir_flag, expiry=None)
        st.session_state.legs.append(new_leg)

# Legs created
def delete_leg(index):
    if 0 <= index < len(st.session_state.legs):
        st.session_state.legs.pop(index)

st.sidebar.markdown("---")
st.sidebar.markdown("### Current Legs:")
if st.session_state.legs:
    for i, leg in enumerate(st.session_state.legs):
        col1, col2 = st.sidebar.columns([4, 1])
        with col1:
            st.write(f"{'Buy' if leg.direction == 1 else 'Sell'} {leg.qty} {leg.type.upper()} @ {leg.K}")
        with col2:
            st.button("X", key=f"del_{i}", on_click=delete_leg, args=(i,))
    if st.sidebar.button("Clear All Legs"):
        st.session_state.legs = []
        st.rerun()
else:
    st.sidebar.info("No legs added yet.")


if not st.session_state.legs:
    st.warning("Please add at least one leg to simulate the strategy.")
    st.stop()

#############
### Main ####
#############

strategy = main.Strategy(st.session_state.legs)
net_premium, leg_prices = strategy.price(spot, T, r, sigma)
S_range = np.linspace(spot * 0.8, spot * 1.2, 200)
pnl_expiration = main.simulate_pnl_exipiration(strategy, S_range, net_premium)
pnl_today = main.simulate_pnl_today(strategy, S_range, T, r, sigma, net_premium)

# Display strategy summary
st.subheader("Strategy summary")
col1, col2, col3= st.columns(3)
max_profit, max_loss = strategy.calculate_max_profit_loss(net_premium)
with col1:
    st.metric("Net Premium", - round(net_premium, 2), border=True)
with col2:
    if max_profit == float('inf'):
        max_profit_text = "Unlimited"
        max_profit_color = "normal"
    else:
        max_profit_text = round(max_profit, 2)
        max_profit_color = "normal"
    st.metric("Max Profit", max_profit_text, border=True)
with col3:
    if max_loss == float('-inf'):
        max_loss_text = "Unlimited"
        max_loss_color = "normal"
    else:
        max_loss_text = - round(max_loss, 2)
        max_loss_color = "normal"
    st.metric("Max Loss", max_loss_text, border=True)

# Display theorical leg prices
st.subheader("Theoretical leg prices")
cols = st.columns(len(leg_prices))
for (k, v), col in zip(leg_prices.items(), cols):
    col.metric(
        label = k,
        value = round(v,2),
        border=True
    )
    
# Display P&L
col1, col2 = st.columns(2)
col1.subheader("P&L Chart")
fig_pl = main.plot_results(S_range, pnl_expiration, pnl_today, spot)
col1.plotly_chart(fig_pl, use_container_width= False)

# Disply 3D surface plot of strategy value
col2.subheader("3D Strategy Value Surface")
T_days_range = np.linspace(T_days, 1, int(T_days))
d3_plot = main.plot_3d_surface(strategy, S_range, T_days_range, r, sigma)
col2.plotly_chart(d3_plot, use_container_width= False)

# Display Greeks
st.subheader("Greeks at Current Spot")
greeks = strategy.greeks(spot, T, r, sigma)
cols = st.columns(len(greeks))
for (k, v), col in zip(greeks.items(), cols):
    with col:
        st.metric(
            label = k.capitalize(),
            value = round(v, 4),
            border=True
        )

greek_plot = main.plot_greeks(strategy, S_range, T, r, sigma)
st.plotly_chart(greek_plot)
