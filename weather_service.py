import json
import pandas as pd
import os


class WeatherService:
    def __init__(self, data_dir="city_data"):
        self.data_dir = data_dir
        self.weather_cache = {}
        self.load_weather_data()

    def load_weather_data(self):
        """Loads weather JSONs and sorts them by date."""
        if not os.path.exists(self.data_dir):
            print(f"WARNING: Directory {self.data_dir} not found.")
            return

        # Load all JSON files in the directory
        files = [f for f in os.listdir(self.data_dir) if f.endswith('.json')]

        print(f"--- Loading Weather Files from {self.data_dir} ---")
        for f in files:
            # Create a clean key (e.g., 'haifa_technion.json' -> 'haifa_technion')
            city_key = f.replace(".json", "").lower()

            try:
                file_path = os.path.join(self.data_dir, f)
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    df = pd.DataFrame(data)

                    # 1. Parse Dates Correctly (Handle Day/Month/Year formats)
                    df['date'] = pd.to_datetime(df['date'], dayfirst=True)

                    # 2. Sort by Date (Oldest -> Newest)
                    df = df.sort_values(by='date')

                    self.weather_cache[city_key] = df
            except Exception as e:
                print(f"Error loading {f}: {e}")

    def get_weather(self, city: str, date_str: str = None):
        """
        Finds weather for a city.
        If date_str is provided (YYYY-MM-DD), finds the closest data point (target 12:00).
        Otherwise, returns the latest data point.
        """
        user_query = city.lower()
        df = None
        matched_key = None

        # --- SMART MATCHING LOGIC ---
        # 1. Iterate through all loaded files to find the best match
        for file_key in self.weather_cache:
            # Normalize file key: "tlv_beach" -> "tlv beach"
            clean_file_key = file_key.replace("_", " ")

            # Check A: Exact match
            if user_query == clean_file_key:
                df = self.weather_cache[file_key]
                matched_key = file_key
                break

            # Check B: Filename is inside User Query (e.g., "tlv beach" inside "tel aviv (tlv beach)")
            if clean_file_key in user_query:
                df = self.weather_cache[file_key]
                matched_key = file_key
                break

            # Check C: User Query is inside Filename (e.g., "haifa" inside "haifa technion")
            if user_query in clean_file_key:
                df = self.weather_cache[file_key]
                matched_key = file_key
                break

        # Fallback: If no match found, use the first available file
        if df is None and len(self.weather_cache) > 0:
            print(f"DEBUG: Could not match city '{city}'. Defaulting to first file.")
            first_key = list(self.weather_cache.keys())[0]
            df = self.weather_cache[first_key]
            matched_key = first_key

        if df is None:
            return "Weather data not available."

        # --- FIND THE CORRECT ROW ---
        if date_str:
            try:
                # Target: Noon on the requested day
                target_date = pd.to_datetime(f"{date_str} 12:00:00")

                # Find the row with the timestamp closest to the target
                closest_idx = (df['date'] - target_date).abs().idxmin()
                selected_row = df.loc[closest_idx]
            except Exception as e:
                print(f"Date parsing error or data missing for date: {e}. Falling back to latest.")
                selected_row = df.iloc[-1]
        else:
            # Default: Latest available data
            selected_row = df.iloc[-1]

        return (
            f"Location: {selected_row.get('sname', 'Unknown')} (File: {matched_key})\n"
            f"Date/Time: {selected_row.get('date')}\n"
            f"Temperature: {selected_row.get('TD')}°C\n"
            f"Humidity: {selected_row.get('RH')}%\n"
            f"Rainfall: {selected_row.get('Rain')} mm\n"
            f"Wind Speed: {selected_row.get('WS')} m/s"
        )