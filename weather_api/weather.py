import requests
import pandas as pd

latitude = 11.2588
longitude = 75.7804

url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": latitude,
    "longitude": longitude,
    "start_date": "2025-01-01",
    "end_date": "2026-08-01",

    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "rain_sum"
    ],

    "timezone": "Asia/Kolkata"
}

response = requests.get(
    url,
    params=params,
    timeout=30
)

response.raise_for_status()

data = response.json()

df = pd.DataFrame({
    "date": data["daily"]["time"],
    "max_temp": data["daily"]["temperature_2m_max"],
    "min_temp": data["daily"]["temperature_2m_min"],
    "precipitation": data["daily"]["precipitation_sum"],
    "rainfall": data["daily"]["rain_sum"]
})

df["date"] = pd.to_datetime(df["date"])

print(df.head())
print(df.tail())

df.to_csv(
    "kozhikode_historical_weather.csv",
    index=False
)