import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Set the title of the app
st.title("Stock Market Data and Technical Analysis")

# Sidebar for user input
st.sidebar.header("User Input")
ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# Fetch stock data
@st.cache_data
def fetch_data(ticker, start, end):
    stock_data = yf.download(ticker, start=start, end=end)
    return stock_data

# Display stock data
data = fetch_data(ticker, start_date, end_date)
st.subheader(f"Stock Data for {ticker}")
st.write(data)

# Plot historical trends
st.subheader("Historical Trends")
fig, ax = plt.subplots()
ax.plot(data.index, data['Close'], label='Close Price')
ax.set_xlabel("Date")
ax.set_ylabel("Close Price")
ax.legend()
st.pyplot(fig)

# Technical Analysis: Moving Averages
st.subheader("Moving Averages")
ma_window = st.sidebar.slider("Moving Average Window", 1, 100, 20)
data[f"MA_{ma_window}"] = data['Close'].rolling(window=ma_window).mean()
fig, ax = plt.subplots()
ax.plot(data.index, data['Close'], label='Close Price')
ax.plot(data.index, data[f"MA_{ma_window}"], label=f"MA_{ma_window}")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)

# Technical Analysis: Bollinger Bands
st.subheader("Bollinger Bands")
bb_window = st.sidebar.slider("Bollinger Bands Window", 1, 100, 20)
data['MA'] = data['Close'].rolling(window=bb_window).mean()
data['BB_up'] = data['MA'] + 2 * data['Close'].rolling(window=bb_window).std()
data['BB_down'] = data['MA'] - 2 * data['Close'].rolling(window=bb_window).std()
fig, ax = plt.subplots()
ax.plot(data.index, data['Close'], label='Close Price')
ax.plot(data.index, data['BB_up'], label='Upper Band')
ax.plot(data.index, data['BB_down'], label='Lower Band')
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)
