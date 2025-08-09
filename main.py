from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
from sklearn.linear_model import LinearRegression
import numpy as np

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sample company list
COMPANIES = [
    {"name": "Apple", "ticker": "AAPL"},
    {"name": "Microsoft", "ticker": "MSFT"},
    {"name": "Google", "ticker": "GOOGL"},
    {"name": "Amazon", "ticker": "AMZN"},
    {"name": "Tesla", "ticker": "TSLA"},
    {"name": "Meta", "ticker": "META"},
    {"name": "NVIDIA", "ticker": "NVDA"},
    {"name": "Netflix", "ticker": "NFLX"},
    {"name": "Adobe", "ticker": "ADBE"},
    {"name": "Intel", "ticker": "INTC"},
    {"name": "Qualcomm", "ticker": "QCOM"},
    {"name": "AMD", "ticker": "AMD"}
]


@app.get("/companies")
def get_companies():
    return COMPANIES

@app.get("/stock/{ticker}")
def get_stock_data(ticker: str):
    try:
        valid_tickers = [c["ticker"] for c in COMPANIES]
        if ticker.upper() not in valid_tickers:
            raise HTTPException(status_code=400, detail="Invalid ticker.")

        stock_data = yf.Ticker(ticker).history(period="30d")
        if stock_data.empty or "Close" not in stock_data.columns:
            raise HTTPException(status_code=404, detail="No valid stock data found.")

        close_prices = stock_data["Close"].dropna()
        dates = close_prices.index.strftime("%Y-%m-%d").tolist()
        prices = close_prices.values.tolist()

        return {
            "ticker": ticker.upper(),
            "dates": dates,
            "prices": prices
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.get("/predict/{ticker}")
def predict_next_day_price(ticker: str):
    try:
        valid_tickers = [c["ticker"] for c in COMPANIES]
        if ticker.upper() not in valid_tickers:
            raise HTTPException(status_code=400, detail="Invalid ticker.")

        stock_data = yf.Ticker(ticker).history(period="60d")
        if stock_data.empty or "Close" not in stock_data.columns:
            raise HTTPException(status_code=404, detail="No valid stock data found.")

        prices = stock_data["Close"].dropna().values
        if len(prices) < 2:
            raise HTTPException(status_code=400, detail="Not enough data for prediction.")

        X = np.arange(len(prices)).reshape(-1, 1)
        y = prices

        model = LinearRegression()
        model.fit(X, y)

        next_day = np.array([[len(prices)]])
        predicted_price = model.predict(next_day)[0]

        return {
            "ticker": ticker.upper(),
            "predicted_price": round(float(predicted_price), 2)
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
