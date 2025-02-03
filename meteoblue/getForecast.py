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
        response.raise_for_status()  # Raise an error for bad HTTP status codes
        data = response.json()
        
        # Uncomment the next line to print the full JSON for debugging:
        # print(json.dumps(data, indent=2))
        
        if "data_day" in data:
            day_data = data["data_day"]
            
            # Retrieve arrays for each parameter
            dates           = day_data.get("time", [])
            temp_means      = day_data.get("temperature_mean", [])
            pressure_means  = day_data.get("sealevelpressure_mean", [])
            wind_means      = day_data.get("windspeed_mean", [])
            humidity_means  = day_data.get("relativehumidity_mean", [])
            
            if dates and temp_means and pressure_means and wind_means and humidity_means:
                print("7-Day Forecast:")
                print("=" * 40)
                for date, temp, pressure, wind, humidity in zip(dates, temp_means, pressure_means, wind_means, humidity_means):
                    print(f"Date: {date}")
                    print(f"  Temperature (mean):       {temp} °C")
                    print(f"  Sea Level Pressure (mean): {pressure} hPa")
                    print(f"  Wind Speed (mean):        {wind} m/s")
                    print(f"  Relative Humidity (mean): {humidity} %")
                    print("-" * 40)
            else:
                print("One or more expected arrays are empty in the forecast data.")
        else:
            print("Unexpected JSON format for forecast data: 'data_day' key not found.")
            print("Full forecast response:")
            print(json.dumps(data, indent=2))
    
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred in forecast request: {http_err} - {response.text}")
    except Exception as err:
        print(f"An error occurred in forecast request: {err}")


def get_current():
    """Retrieve and display the current weather data."""
    url = "https://my.meteoblue.com/packages/basic-day_current"
    
    # Parameters for the current weather request
    params = {
        "apikey": "2RmKPQgb9fUSAvIM",  # your API key
        "lat": 40.4165,                # latitude for Madrid, Spain
        "lon": -3.70256,               # longitude for Madrid, Spain
        "asl": 665,                    # altitude above sea level in meters
        "format": "json"
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Uncomment the next line to print the full JSON for debugging:
        # print(json.dumps(data, indent=2))
        
        # The current data may be wrapped under a key; often it is "data_current".
        # If not, assume the keys are at the top level.
        if "data_current" in data:
            current = data["data_current"]
        else:
            current = data
        
        # Retrieve current weather parameters.
        temperature        = current.get("temperature", "N/A")
        windspeed          = current.get("windspeed", "N/A")
        pictocode          = current.get("pictocode", "N/A")
        pictocode_detailed = current.get("pictocode_detailed", "N/A")
        is_daylight        = current.get("is_daylight", "N/A")
        is_observation     = current.get("is_observation", "N/A")
        zenith_angle       = current.get("zenith_angle", "N/A")
        metar_id           = current.get("metar_id", "N/A")
        
        print("\nCurrent Weather:")
        print("=" * 40)
        print(f"Temperature:            {temperature} °C")
        print(f"Wind Speed:             {windspeed} m/s")
        print(f"Pictocode:              {pictocode}")
        print(f"Pictocode (Detailed):   {pictocode_detailed}")
        print(f"Is Daylight:            {is_daylight}")
        print(f"Is Observation:         {is_observation}")
        print(f"Zenith Angle:           {zenith_angle}°")
        print(f"METAR ID:               {metar_id}")
        print("=" * 40)
    
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred in current weather request: {http_err} - {response.text}")
    except Exception as err:
        print(f"An error occurred in current weather request: {err}")


if __name__ == "__main__":
    get_forecast()
    get_current()
