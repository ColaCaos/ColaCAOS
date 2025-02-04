import requests
import json

def get_forecast():
    """Retrieve and display the 7-day forecast data."""
    url = "https://my.meteoblue.com/packages/basic-day"
    
    # Parameters for the forecast request (using Madrid coordinates as example)
    params = {
        "apikey": "2RmKPQgb9fUSAvIM",  # your API key
        "lat": 40.4165,                # latitude for Madrid, Spain
        "lon": -3.70256,               # longitude for Madrid, Spain
        "asl": 665,                    # altitude above sea level in meters
        "format": "json"
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()
    except Exception as e:
        print("Error retrieving forecast data:", e)
        return

    # The forecast data is located under the "data_day" key.
    forecast = data.get("data_day")
    if not forecast:
        print("No forecast data found in the response.")
        return
    
    # Extract forecast fields (lists)
    dates = forecast.get("time", [])
    temperature_instant = forecast.get("temperature_instant", [])
    temperature_max = forecast.get("temperature_max", [])
    temperature_min = forecast.get("temperature_min", [])
    precipitation = forecast.get("precipitation", [])
    windspeed_mean = forecast.get("windspeed_mean", [])
    relativehumidity_mean = forecast.get("relativehumidity_mean", [])
    windspeed_max = forecast.get("windspeed_max", [])
    
    print("7-Day Forecast for Madrid, Spain:")
    print("=" * 40)
    
    # Iterate over each day (assuming all lists are aligned)
    for i, date in enumerate(dates):
        print(f"Date: {date}")
        if i < len(temperature_instant):
            print(f"  Instant Temperature: {temperature_instant[i]}°C")
        if i < len(temperature_max):
            print(f"  Max Temperature: {temperature_max[i]}°C")
        if i < len(temperature_min):
            print(f"  Min Temperature: {temperature_min[i]}°C")
        if i < len(precipitation):
            print(f"  Precipitation: {precipitation[i]} mm")
        if i < len(windspeed_mean):
            print(f"  Mean Wind Speed: {windspeed_mean[i]} m/s")
        if i < len(relativehumidity_mean):
            print(f"  Mean Humidity: {relativehumidity_mean[i]}%")
        if i < len(windspeed_max):
            print(f"  Max Wind Speed: {windspeed_max[i]} m/s")
        print("-" * 40)

if __name__ == "__main__":
    get_forecast()
