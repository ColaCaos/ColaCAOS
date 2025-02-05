import requests
import csv
import os
import json

def get_forecast():
    """Retrieve and display the 7-day forecast data and save each day to a CSV file."""
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

    # The forecast data is under the "data_day" key.
    forecast = data.get("data_day")
    if not forecast:
        print("No forecast data found in the response.")
        return
    
    # Extract the fields we want from the JSON response
    dates = forecast.get("time", [])
    temperature_instant = forecast.get("temperature_instant", [])
    temperature_mean = forecast.get("temperature_mean", [])
    temperature_max = forecast.get("temperature_max", [])
    temperature_min = forecast.get("temperature_min", [])
    precipitation = forecast.get("precipitation", [])
    windspeed_mean = forecast.get("windspeed_mean", [])
    windspeed_max = forecast.get("windspeed_max", [])
    relativehumidity_mean = forecast.get("relativehumidity_mean", [])
    
    print("7-Day Forecast for Madrid, Spain:")
    print("=" * 40)
    
    # Prepare CSV file
    csv_filename = "forecast_madrid.csv"
    file_exists = os.path.isfile(csv_filename)
    
    with open(csv_filename, "a", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "date",
            "temperature_instant",
            "temperature_mean",
            "temperature_max",
            "temperature_min",
            "precipitation",
            "windspeed_mean",
            "windspeed_max",
            "relativehumidity_mean"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header if the file did not exist
        if not file_exists:
            writer.writeheader()
        
        # Iterate over each day (assuming all lists are aligned)
        for i, date in enumerate(dates):
            # Display the forecast on the console
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
            
            # Write a line to the CSV file
            writer.writerow({
                "date": date,
                "temperature_instant": temperature_instant[i] if i < len(temperature_instant) else "",
                "temperature_mean": temperature_mean[i] if i < len(temperature_mean) else "",
                "temperature_max": temperature_max[i] if i < len(temperature_max) else "",
                "temperature_min": temperature_min[i] if i < len(temperature_min) else "",
                "precipitation": precipitation[i] if i < len(precipitation) else "",
                "windspeed_mean": windspeed_mean[i] if i < len(windspeed_mean) else "",
                "windspeed_max": windspeed_max[i] if i < len(windspeed_max) else "",
                "relativehumidity_mean": relativehumidity_mean[i] if i < len(relativehumidity_mean) else ""
            })

    print(f"\nForecast data saved to '{csv_filename}'.")

if __name__ == "__main__":
    get_forecast()
