import streamlit as st

# Set the title of the app
st.title("Stock Market Data and Technical Analysis")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Stock Market Data"])

# Routing logic
if page == "Stock Market Data":
    import pages.stock_market_data
