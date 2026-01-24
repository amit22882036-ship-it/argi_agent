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

        files = [f for f in os.listdir(self.data_dir) if f.endswith('.json')]

        print(f"--- Loading Weather Files from {self.data_dir} ---")
        for f in files:
            # Clean key: 'haifa_technion.json' -> 'haifa_technion'
            city_key = f.replace(".json", "").lower()

            try:
                file_path = os.path.join(self.data_dir, f)
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    df = pd.DataFrame(data)

                    # Parse and Sort
                    df['date'] = pd.to_datetime(df['date'], dayfirst=True)
                    df = df.sort_values(by='date')

                    self.weather_cache[city_key] = df
            except Exception as e:
                print(f"Error loading {f}: {e}")

    def get_current_weather(self, city: str):
        """
        Finds the weather file for the city using smart matching.
        """
        # 1. Normalize User Input (e.g., "Tel Aviv (TLV Beach)" -> "tel aviv (tlv beach)")
        user_query = city.lower()

        df = None
        matched_key = None

        # 2. Iterate through all loaded files to find the best match
        for file_key in self.weather_cache:
            # Normalize file key: "tlv_beach" -> "tlv beach"
            clean_file_key = file_key.replace("_", " ")

            # MATCHING LOGIC:
            # A. Exact Match
            if user_query == clean_file_key:
                df = self.weather_cache[file_key]
                matched_key = file_key
                break

            # B. Filename is inside User Query
            # (e.g., "tlv beach" is inside "tel aviv (tlv beach)") -> THIS FIXES YOUR ISSUE
            if clean_file_key in user_query:
                df = self.weather_cache[file_key]
                matched_key = file_key
                break

            # C. User Query is inside Filename
            # (e.g., "haifa" is inside "haifa technion")
            if user_query in clean_file_key:
                df = self.weather_cache[file_key]
                matched_key = file_key
                break

        # 3. Fallback: If no match found, warn and use the first available file
        if df is None and len(self.weather_cache) > 0:
            print(f"DEBUG: Could not match city '{city}'. Defaulting to first file.")
            first_key = list(self.weather_cache.keys())[0]
            df = self.weather_cache[first_key]
            matched_key = first_key

        if df is None:
            return "Weather data not available."

        # 4. Get the LATEST row
        latest = df.iloc[-1]

        return (
            f"Location: {latest.get('sname', 'Unknown')} (File: {matched_key})\n"
            f"Date/Time: {latest.get('date')}\n"
            f"Temperature: {latest.get('TD')}°C\n"
            f"Humidity: {latest.get('RH')}%\n"
            f"Rainfall: {latest.get('Rain')} mm\n"
            f"Wind Speed: {latest.get('WS')} m/s"
        )