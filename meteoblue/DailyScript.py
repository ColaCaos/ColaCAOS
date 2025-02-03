import requests
import csv
import os
import json
from datetime import datetime, timedelta

# --- Configuration ---
API_KEY = "DUN6M3QJ4X5G2B76GVFB7GH5D"
location = "Madrid,Spain"  # Note: No spaces; Visual Crossing accepts "Madrid,Spain"

# We will work with four parameters: temperature, humidity, pressure, wind speed.
# For each, we define:
#   - API field (the key in the JSON response)
#   - Forecast CSV file name
#   - Retro CSV file name
PARAMETERS = [
    ("temperature", "temp", "temperatureForecast.csv", "temperatureRetro.csv"),
    ("humidity", "humidity", "humidityForecast.csv", "humidityRetro.csv"),
    ("pressure", "pressure", "pressureForecast.csv", "pressureRetro.csv"),
    ("wind", "windspeed", "windForecast.csv", "windRetro.csv"),
]

# --- Helper Functions ---

def format_date(dt):
    """
    Returns date in American notation (M-D-YYYY), e.g. "2-1-2025".
    """
    return f"{dt.month}-{dt.day}-{dt.year}"

def append_row_to_csv(filename, row):
    """
    Appends a row (a list) to the CSV file named filename.
    If the file does not exist, it will be created.
    """
    file_exists = os.path.isfile(filename)
    with open(filename, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # (Optionally, you could add a header if the file is new.)
        writer.writerow(row)

def read_forecast_csv(filename):
    """
    Reads the CSV file (if it exists) and returns a dictionary mapping the
    forecast creation date (first column) to the entire row (list of strings).
    """
    data = {}
    if os.path.isfile(filename):
        with open(filename, newline="") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row:
                    # row[0] is the forecast creation date (in American format)
                    data[row[0]] = row
    return data

def get_timeline_data(start_date_str, end_date_str):
    """
    Makes a request to Visual Crossing's timeline endpoint for the given date range.
    Returns the JSON response.
    """
    url = (
        f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"
        f"{location}/{start_date_str}/{end_date_str}"
        f"?unitGroup=metric&include=days&key={API_KEY}&contentType=json"
    )
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def get_forecast_data():
    """
    Gets forecast data for 15 days (today + 14 days ahead) using Visual Crossing.
    Returns the list of daily forecasts from the "days" key.
    """
    today = datetime.today()
    start_date = today
    end_date = today + timedelta(days=14)
    start_str = today.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    data = get_timeline_data(start_str, end_str)
    return data.get("days", [])

def get_historical_data(target_date):
    """
    Gets historical data for a single day (target_date).
    target_date: datetime object.
    Returns the day dictionary (if available) from the API response.
    """
    date_str = target_date.strftime("%Y-%m-%d")
    data = get_timeline_data(date_str, date_str)
    days = data.get("days", [])
    if days:
        return days[0]
    return {}

# --- Main Script Functions ---

def record_forecasts():
    """
    Retrieves forecast data from Visual Crossing (today + 14 days ahead)
    and appends a new row to each forecast CSV file.
    
    Each row has 15 columns:
      Column 1: Today's date (forecast creation date)
      Columns 2-15: Forecast values for 1 day ahead, 2 days ahead, …, 14 days ahead.
    """
    forecast_days = get_forecast_data()  # list of dicts; index 0 is today's forecast, index 1 tomorrow, etc.
    today = datetime.today()
    today_str = format_date(today)
    
    # For each parameter, build the row from the forecast data.
    # We use forecast_days[1:15] so that column 2 corresponds to "tomorrow" (1 day ahead) and so on.
    for param_name, api_field, forecast_file, _ in PARAMETERS:
        # Ensure that we have at least 15 forecast entries.
        if len(forecast_days) < 15:
            # If not, pad with empty strings.
            forecasts = [day.get(api_field, "") for day in forecast_days[1:]] + [""] * (14 - (len(forecast_days)-1))
        else:
            forecasts = [day.get(api_field, "") for day in forecast_days[1:15]]
        row = [today_str] + forecasts  # Total of 1 + 14 = 15 columns.
        append_row_to_csv(forecast_file, row)
        print(f"Appended forecast row for {param_name} to {forecast_file}")

def record_retro():
    """
    Retrieves yesterday's historical (observed) data and then builds a retro row
    for each parameter by combining the historical value with forecasts that were made in the past.
    
    Retro CSV row format (15 columns + 1 initial column = 16 columns total):
      Column 1: Yesterday's date (target date) in American notation.
      Column 2: Historical observed value for that date.
      Columns 3-16: For each offset n = 1 to 14, the forecast for the target date
                    that was made n days before the target date.
                    
    For example, if yesterday is 2-1-2025:
      - f1 (column 3) is taken from the forecast CSV row whose date is (2-1-2025 - 1 day = 1-31-2025),
        and from that row, the forecast for 1 day ahead (column 2).
      - f2 (column 4) is from the forecast CSV row for 1-30-2025 (forecast made 2 days before),
        from column 3, and so on.
    """
    yesterday = datetime.today() - timedelta(days=1)
    yesterday_str = format_date(yesterday)
    
    # Get historical data for yesterday.
    hist_day = get_historical_data(yesterday)
    
    for param_name, api_field, forecast_file, retro_file in PARAMETERS:
        # Historical observed value for yesterday.
        hist_value = hist_day.get(api_field, "")
        # Start the retro row with yesterday's date and the historical value.
        retro_row = [yesterday_str, hist_value]
        
        # Read the forecast CSV file for this parameter.
        forecast_data = read_forecast_csv(forecast_file)
        
        # For each offset from 1 to 14:
        #   target forecast for yesterday, made n days before, is stored in the forecast CSV row
        #   whose first column is (yesterday - n days) and in that row, the forecast value is in column index = n.
        #   (Because column 1 is the creation date, column 2 is the forecast for 1 day ahead, etc.)
        for offset in range(1, 15):
            forecast_date = yesterday - timedelta(days=offset)
            forecast_date_str = format_date(forecast_date)
            # Get the row from the forecast CSV file.
            if forecast_date_str in forecast_data:
                row = forecast_data[forecast_date_str]
                # Check if the row has enough columns (we expect 15 columns total).
                # The forecast value we want is at index = offset (since index 1 corresponds to forecast 1 day ahead).
                if len(row) > offset:
                    forecast_value = row[offset]
                else:
                    forecast_value = ""
            else:
                forecast_value = ""
            retro_row.append(forecast_value)
        
        append_row_to_csv(retro_file, retro_row)
        print(f"Appended retro row for {param_name} to {retro_file}")

def main():
    # Part 1: Record today's forecast data.
    record_forecasts()
    # Part 2: Record yesterday's historical data and build retro rows.
    record_retro()

if __name__ == "__main__":
    main()
