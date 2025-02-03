import requests
import json
from datetime import datetime, timedelta

# Your Visual Crossing API key
API_KEY = "DUN6M3QJ4X5G2B76GVFB7GH5D"

# Location for which you want the forecast (Madrid, Spain)
location = "Madrid,Spain"

# Set the forecast date range:
# Start date: today
# End date: 13 days from today (for a total of 14 days)
start_date = datetime.today()
end_date = start_date + timedelta(days=13)

# Format dates as YYYY-MM-DD (Visual Crossing requires this format)
start_date_str = start_date.strftime("%Y-%m-%d")
end_date_str = end_date.strftime("%Y-%m-%d")

# Build the API URL.
# - unitGroup=metric: returns values in metric units (°C, mb, km/h, etc.)
# - include=days: includes daily forecast details.
# - contentType=json: returns the response in JSON format.
url = (
    f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"
    f"{location}/{start_date_str}/{end_date_str}"
    f"?unitGroup=metric&include=days&key={API_KEY}&contentType=json"
)

def get_forecast():
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad status codes
        data = response.json()

        # Check for the expected "days" key that contains daily forecast data.
        if "days" in data:
            days = data["days"]
            print(f"14-Day Forecast for {location} from {start_date_str} to {end_date_str}:")
            print("=" * 50)
            for day in days:
                date = day.get("datetime", "N/A")
                # "temp" is the daily average temperature.
                avg_temp = day.get("temp", "N/A")
                # "pressure" is in millibars.
                pressure = day.get("pressure", "N/A")
                # "humidity" is the relative humidity (in percent).
                humidity = day.get("humidity", "N/A")
                # "windspeed" is in km/h (when using metric units).
                wind_speed = day.get("windspeed", "N/A")

                print(f"Date: {date}")
                print(f"  Mean Temperature: {avg_temp} °C")
                print(f"  Pressure:         {pressure} mb")
                print(f"  Humidity:         {humidity} %")
                print(f"  Wind Speed:       {wind_speed} km/h")
                print("-" * 50)
        else:
            print("The expected 'days' key was not found in the response:")
            print(json.dumps(data, indent=2))
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err} - {response.text}")
    except Exception as err:
        print(f"An error occurred: {err}")

if __name__ == "__main__":
    get_forecast()
