import pandas as pd
import os


# First here there are Normal things going onm, latter if we find someting intersting them we will add
def prepare_data(ticker):

    file_path = f"DSM-9/data/{ticker}_prices.csv"

    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found. Run ingest.py first!")
        return

    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df.index)

    # --------------------# Calculating Some SMA #-------------------#
    df['MA_7'] = df['Close'].rolling(window=7).mean()
    df['MA_30'] = df['Close'].rolling(window=30).mean()

    df['Daily_Return'] = df['Close'].pct_change()

    df.dropna(inplace=True)

    print(
        f"Processed data for {ticker}. New Clues added: MA_7, MA_30, Daily_Return")
    print(df[['Close', 'MA_7', 'MA_30']].tail())

    df.to_csv(f"data/{ticker}_processed.csv", index=False)


if __name__ == "__main__":
    prepare_data("NVDA")  # Or whatever stock you managed to download
