import yfinance as yf
import streamlit as st
from PIL import Image
from urllib.request import urlopen

# Function to fetch cryptocurrency data
def get_crypto_data(symbol):
    crypto_data = yf.Ticker(symbol)
    return crypto_data.history(period="max")

# Function to add an icon to the dashboard
def add_icon(image_url):
    return Image.open(urlopen(image_url))

# Function to calculate moving averages
def calculate_moving_average(data, window_size):
    data['MA'] = data['Close'].rolling(window=window_size).mean()
    return data

# Function to calculate daily percentage change
def calculate_daily_percentage_change(data):
    data['Daily_Return'] = data['Close'].pct_change() * 100
    return data

# Function to train and predict using a linear regression model
def train_linear_regression_model(data):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression

    X = data.index.values.astype(int).reshape(-1, 1)
    y = data['Close'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    return predictions

# Cryptocurrency symbols
cryptos = {
    'Bitcoin': 'BTC-USD',
    'Ethereum': 'ETH-USD',
    'Ripple': 'XRP-USD',
    'Bitcoin_Cash': 'BCH-USD'
}

# Streamlit App
st.title("Cryptocurrency Daily Prices ₿")
st.header("Your Dashboard ✨")
st.subheader("You can add more crypto in the code </>")

for crypto_name, crypto_symbol in cryptos.items():
    st.write(f"## {crypto_name} ($)")
    
    # Adding icon for the cryptocurrency
    crypto_icon_url = f'https://s2.coinmarketcap.com/static/img/coins/64x64/{yf.Ticker(crypto_symbol).info["id"]}.png'
    image_crypto = add_icon(crypto_icon_url)
    st.image(image_crypto, use_column_width=False)
    
    # Fetch cryptocurrency data
    crypto_data = get_crypto_data(crypto_symbol)
    
    # Line chart for Close Price history
    st.line_chart(crypto_data.Close, use_container_width=True)
    
    # Interactive Date Range Selector
    date_range = st.date_input(f"Select Date Range for {crypto_name}", [crypto_data.index.min(), crypto_data.index.max()])
    selected_data = crypto_data.loc[date_range[0]:date_range[1]]
    st.line_chart(selected_data.Close, use_container_width=True)
    
    # Moving Averages
    st.write(f"### {crypto_name} Moving Averages")
    window_size = st.slider(f"Select Moving Average Window Size for {crypto_name}", min_value=1, max_value=365, value=30)
    crypto_data_with_ma = calculate_moving_average(crypto_data.copy(), window_size)
    st.line_chart(crypto_data_with_ma[['Close', 'MA']], use_container_width=True)
    
    # Daily Percentage Change
    st.write(f"### {crypto_name} Daily Percentage Change")
    crypto_data_with_percentage_change = calculate_daily_percentage_change(crypto_data.copy())
    st.line_chart(crypto_data_with_percentage_change['Daily_Return'], use_container_width=True)

    # Linear Regression Model and Prediction
    st.write(f"### {crypto_name} Price Prediction")
    predictions = train_linear_regression_model(crypto_data.copy())
    st.line_chart(predictions, use_container_width=True)
