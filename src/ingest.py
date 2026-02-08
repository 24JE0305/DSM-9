import yfinance as yf
import os
import time

# This will not work I guess but don't worry we will make our self by google colab


def download_data(ticker, period="1y"):
    if not os.path.exists('data'):
        os.makedirs('data')

    print(f"Fetching data for {ticker}...")

    # NEW: Create a 'Ticker' object first. This is more stable.
    stock = yf.Ticker(ticker)

    # NEW: Use a longer 'timeout' and 'proxy' settings if needed,
    # but for now, we'll just use the history method.
    stock_data = stock.history(period=period)

    if stock_data.empty:
        print(
            f"❌ Failed to get data for {ticker}. Yahoo might be blocking us.")
        return

    file_path = f"DSm-9/data/{ticker}_prices.csv"
    stock_data.to_csv(file_path)

    print(f"✅ Success! Data saved to {file_path}")
    print(stock_data.tail())


if __name__ == "__main__":
    download_data("AAPL")
