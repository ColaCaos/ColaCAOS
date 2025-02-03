import requests
import json
from datetime import datetime, timedelta

# Your Visual Crossing API key
API_KEY = "DUN6M3QJ4X5G2B76GVFB7GH5D"

# Location for which you want the data (Madrid, Spain)
location = "Madrid,Spain"

# Calculate the date range: past 30 days until today.
end_date = datetime.today()
start_date = end_date - timedelta(days=30)

# Format dates as YYYY-MM-DD (Visual Crossing requires this format)
start_date_str = start_date.strftime("%Y-%m-%d")
end_date_str = end_date.strftime("%Y-%m-%d")

# Build the API URL.
# 'unitGroup=metric' returns temperatures in Celsius, pressure in millibars, and wind speed in km/h.
# 'include=days' ensures we get daily data.
url = (
    f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"
    f"{location}/{start_date_str}/{end_date_str}"
    f"?unitGroup=metric&include=days&key={API_KEY}&contentType=json"
)

def get_historical_weather():
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        # Check that we have daily data under the "days" key.
        if "days" in data:
            days = data["days"]
            print(f"Historical weather for {location} from {start_date_str} to {end_date_str}:")
            print("=" * 50)
            for day in days:
                date = day.get("datetime", "N/A")
                # Visual Crossing returns "temp" as the average temperature.
                avg_temp = day.get("temp", "N/A")
                # Pressure field is returned as "pressure" (in millibars)
                pressure = day.get("pressure", "N/A")
                # Humidity is returned as a percentage.
                humidity = day.get("humidity", "N/A")
                # Wind speed is returned as "windspeed" (in km/h when using metric)
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
    get_historical_weather()
