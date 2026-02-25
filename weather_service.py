import json
import pandas as pd
import os
import re


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
            city_key = f.replace(".json", "").lower()
            try:
                file_path = os.path.join(self.data_dir, f)
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    df = pd.DataFrame(data)

                    # 1. המרה לתאריך - וביטול אזורי זמן כדי למנוע קריסות בהשוואות!
                    df['date'] = pd.to_datetime(df['date'], dayfirst=True).dt.tz_localize(None)

                    # 2. מיון התאריכים (מהישן לחדש)
                    df = df.sort_values(by='date')

                    # איפוס אינדקס כדי שהחיפוש יעבוד חלק
                    df = df.reset_index(drop=True)

                    self.weather_cache[city_key] = df
            except Exception as e:
                print(f"Error loading {f}: {e}")

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
        df = None
        matched_key = None

        # --- בחירת הקובץ המתאים ---
        for file_key in self.weather_cache:
            clean_file_key = file_key.replace("_", " ")
            if user_query == clean_file_key or clean_file_key in user_query or user_query in clean_file_key:
                df = self.weather_cache[file_key]
                matched_key = file_key
                break

        if df is None and len(self.weather_cache) > 0:
            first_key = list(self.weather_cache.keys())[0]
            df = self.weather_cache[first_key]
            matched_key = first_key

        if df is None:
            return "Error: No weather data available."

        # --- מציאת השורה של התאריך המבוקש ---
        if date_str:
            try:
                # יצירת תאריך מטרה (ללא אזור זמן) לשעה 12:00 בצהריים
                target_date = pd.to_datetime(f"{date_str} 12:00:00").tz_localize(None)

                # מציאת האינדקס של השורה עם התאריך הכי קרוב
                closest_idx = (df['date'] - target_date).abs().idxmin()
                selected_row = df.loc[closest_idx]

                print(f"[SUCCESS] Matched Requested Date: {date_str} -> Found Data Date: {selected_row['date']}")
            except Exception as e:
                print(f"[CRITICAL ERROR] Failed to calculate closest date: {e}")
                selected_row = df.iloc[-1]
        else:
            print("[WARNING] No date found in query, using last row.")
            selected_row = df.iloc[-1]

        # החזרת הנתונים לסוכן בצורה שתכריח אותו להבין שזה "היום"
        return (
            f"!!! CRITICAL INSTRUCTION: TREAT THIS SPECIFIC DATE AS 'TODAY' !!!\n"
            f"Location: {matched_key}\n"
            f"Weather Data Date: {selected_row.get('date')}\n"
            f"Temperature: {selected_row.get('TD')}°C\n"
            f"Humidity: {selected_row.get('RH')}%\n"
            f"Rainfall: {selected_row.get('Rain')} mm\n"
        )