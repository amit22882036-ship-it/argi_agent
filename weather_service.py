import gc
import json
import pandas as pd
import os
import re


class WeatherService:
    def __init__(self, data_dir="city_data"):
        self.data_dir = data_dir
        # Cache only the resolved row (city_key, date_str) -> row dict — not the full DataFrame
        self._row_cache = {}
        if os.path.exists(data_dir):
            self._available = {f.replace(".json", "").lower(): f
                               for f in os.listdir(data_dir) if f.endswith('.json')}
            print(f"--- Weather Service ready: {len(self._available)} cities indexed ---")
        else:
            self._available = {}
            print(f"WARNING: Directory {data_dir} not found.")

    def _find_row(self, city_key: str, date_str: str):
        """
        Load the city file, find the closest row to date_str, cache the row dict,
        then immediately discard the full DataFrame to keep memory flat.
        """
        cache_key = (city_key, date_str)
        if cache_key in self._row_cache:
            return self._row_cache[cache_key], city_key

        filename = self._available.get(city_key)
        if not filename:
            return None, city_key

        try:
            file_path = os.path.join(self.data_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as fh:
                data = json.load(fh)

            df = pd.DataFrame(data)
            del data
            gc.collect()

            df['date'] = pd.to_datetime(df['date'], dayfirst=True).dt.tz_localize(None)

            if date_str:
                target = pd.to_datetime(f"{date_str} 12:00:00").tz_localize(None)
                idx = (df['date'] - target).abs().idxmin()
                row = df.loc[idx].to_dict()
                print(f"[SUCCESS] {city_key}: requested {date_str} → matched {row.get('date')}")
            else:
                row = df.iloc[-1].to_dict()
                print(f"[WARNING] No date in query, using last row for {city_key}")

            del df
            gc.collect()

            self._row_cache[cache_key] = row
            return row, city_key

        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return None, city_key

    def get_weather(self, query: str):
        """
        מחפש את העיר ואת התאריך מתוך המחרוזת של הסוכן.
        """
        print(f"\n[DEBUG] Weather Tool Called -> Query: {query}")

        # --- חילוץ התאריך ---
        date_match = re.search(r'\d{4}-\d{2}-\d{2}', query)
        date_str = date_match.group(0) if date_match else None

        # --- חילוץ העיר ---
        if date_str:
            city = query.replace(date_str, "").replace(" on ", "").strip()
        else:
            city = query.strip()

        # ניקוי נוסף אם LangChain שלח מילון טקסטואלי בטעות
        if "{" in city:
            import ast
            try:
                parsed = ast.literal_eval(city)
                if isinstance(parsed, dict):
                    city = list(parsed.values())[0]
            except:
                pass

        user_query = city.lower().strip()
        matched_key = None

        # --- בחירת הקובץ המתאים ---
        for file_key in self._available:
            clean_file_key = file_key.replace("_", " ")
            if user_query == clean_file_key or clean_file_key in user_query or user_query in clean_file_key:
                matched_key = file_key
                break

        if matched_key is None and len(self._available) > 0:
            matched_key = list(self._available.keys())[0]

        if matched_key is None:
            return "Error: No weather data available."

        row, matched_key = self._find_row(matched_key, date_str)

        if row is None:
            return "Error: Could not load weather data."

        # Build response with all available columns
        lines = [
            "!!! CRITICAL INSTRUCTION: TREAT THIS SPECIFIC DATE AS 'TODAY' !!!",
            f"Location: {matched_key}",
            f"Weather Data Date: {row.get('date')}",
        ]
        if row.get('TD') is not None:
            lines.append(f"Temperature: {row.get('TD')}°C")
        if row.get('RH') is not None:
            lines.append(f"Humidity: {row.get('RH')}%")
        if row.get('Rain') is not None:
            lines.append(f"Rainfall: {row.get('Rain')} mm")
        # Include all remaining columns (skip already-formatted ones and empty/dash values)
        skip = {'date', 'TD', 'RH', 'Rain'}
        for k, v in row.items():
            if k in skip:
                continue
            if v is None or v == '-' or (isinstance(v, float) and pd.isna(v)):
                continue
            lines.append(f"{k}: {v}")

        return "\n".join(lines)
