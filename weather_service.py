import gc
import json
import pandas as pd
import os
import re


class WeatherService:
    def __init__(self, data_dir="city_data"):
        self.data_dir = data_dir
        self._daily_cache = {}  # city_key -> compact daily DataFrame (one row per day)
        if os.path.exists(data_dir):
            self._available = {f.replace(".json", "").lower(): f
                               for f in os.listdir(data_dir) if f.endswith('.json')}
            print(f"--- Weather Service ready: {len(self._available)} cities indexed ---")
        else:
            self._available = {}
            print(f"WARNING: Directory {data_dir} not found.")

    def _get_daily(self, city_key: str) -> pd.DataFrame | None:
        """
        Load city JSON once, aggregate to one row per day, cache the compact result.
        First call: ~1-2s (loads + aggregates 13-16MB file).
        All subsequent calls for same city: <1ms from cache.
        """
        if city_key in self._daily_cache:
            return self._daily_cache[city_key]

        filename = self._available.get(city_key)
        if not filename:
            return None

        try:
            file_path = os.path.join(self.data_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as fh:
                data = json.load(fh)

            df = pd.DataFrame(data)
            del data
            gc.collect()

            df['date'] = pd.to_datetime(df['date'], dayfirst=True).dt.tz_localize(None)
            df['_day'] = df['date'].dt.normalize()

            # Coerce all non-date columns to numeric (converts '-' / garbage → NaN)
            for c in df.columns:
                if c not in ('date', '_day', 'sname'):
                    df[c] = pd.to_numeric(df[c], errors='coerce')

            # Build named aggregation dynamically from available numeric columns
            agg = {}
            cols = set(df.columns)

            if 'TD'   in cols: agg.update({'TD_avg': ('TD', 'mean'), 'TD_max': ('TD', 'max'), 'TD_min': ('TD', 'min')})
            if 'RH'   in cols: agg['RH_avg']       = ('RH',   'mean')
            if 'Rain' in cols: agg['Rain_total']    = ('Rain', 'sum')
            if 'WS'   in cols: agg.update({'WS_avg': ('WS', 'mean'), 'WS_max': ('WS', 'max')})
            if 'WD'   in cols: agg['WD_avg']        = ('WD',   'mean')
            if 'STDwd' in cols: agg['STDwd_avg']    = ('STDwd','mean')

            # Any remaining numeric cols → daily mean
            skip = {'TD', 'RH', 'Rain', 'WS', 'WD', 'STDwd', 'Time'}
            for c in cols - skip:
                if c.startswith('_') or c in ('date', 'sname'):
                    continue
                if pd.api.types.is_numeric_dtype(df[c]) and f"{c}_avg" not in agg:
                    agg[f"{c}_avg"] = (c, 'mean')

            daily = df.groupby('_day').agg(**agg).round(2).reset_index()
            daily.rename(columns={'_day': 'date'}, inplace=True)

            del df
            gc.collect()

            self._daily_cache[city_key] = daily
            print(f"[Weather] {city_key}: cached {len(daily)} days")
            return daily

        except Exception as e:
            print(f"[Weather] Error loading {filename}: {e}")
            return None

    # ------------------------------------------------------------------
    def get_weather(self, query: str) -> str:
        """
        Parse city + date from agent query string, return rich context:
        today's conditions + 7-day + 30-day summaries for agricultural planning.
        """
        print(f"\n[DEBUG] Weather Tool Called -> Query: {query}")

        # Extract date
        date_match = re.search(r'\d{4}-\d{2}-\d{2}', query)
        date_str = date_match.group(0) if date_match else None

        # Extract city
        city = query.replace(date_str, "").replace(" on ", "").strip() if date_str else query.strip()

        if "{" in city:
            import ast
            try:
                parsed = ast.literal_eval(city)
                if isinstance(parsed, dict):
                    city = list(parsed.values())[0]
            except Exception:
                pass

        user_query = city.lower().strip()

        # Match city to available files
        matched_key = None
        for file_key in self._available:
            clean = file_key.replace("_", " ")
            if user_query == clean or clean in user_query or user_query in clean:
                matched_key = file_key
                break
        if matched_key is None and self._available:
            matched_key = next(iter(self._available))

        if matched_key is None:
            return "Error: No weather data available."

        daily = self._get_daily(matched_key)
        if daily is None or daily.empty:
            return "Error: Could not load weather data."

        # Target date
        target = pd.to_datetime(date_str) if date_str else daily['date'].iloc[-1]

        # Find closest available day (sub-millisecond on cached daily DataFrame)
        idx = (daily['date'] - target).abs().idxmin()
        today = daily.loc[idx]
        actual_date = today['date'].strftime('%Y-%m-%d')
        print(f"[Weather] {matched_key}: requested {date_str} → matched {actual_date}")

        # 7-day and 30-day windows (days strictly before target)
        prev7  = daily[(daily['date'] >= target - pd.Timedelta(days=7))  & (daily['date'] < target)]
        prev30 = daily[(daily['date'] >= target - pd.Timedelta(days=30)) & (daily['date'] < target)]

        def _val(row, key, unit=""):
            v = row.get(key)
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return "N/A"
            return f"{v}{unit}"

        lines = [
            "!!! CRITICAL INSTRUCTION: TREAT THIS DATE AS 'TODAY' FOR ALL ADVICE !!!",
            f"Location: {matched_key}  |  Reference date: {actual_date}",
            "",
            "── TODAY'S CONDITIONS ──────────────────────────────",
            f"  Temperature : avg {_val(today,'TD_avg','°C')}  max {_val(today,'TD_max','°C')}  min {_val(today,'TD_min','°C')}",
            f"  Humidity    : {_val(today,'RH_avg','%')}",
            f"  Rainfall    : {_val(today,'Rain_total',' mm')}",
            f"  Wind        : avg {_val(today,'WS_avg',' m/s')}  max {_val(today,'WS_max',' m/s')}  direction {_val(today,'WD_avg','°')}",
        ]

        if len(prev7) > 0:
            n7 = len(prev7)
            frost7 = int((prev7['TD_min'] < 0).sum()) if 'TD_min' in prev7.columns else 'N/A'
            lines += [
                "",
                f"── PAST {n7} DAYS (up to {actual_date}) ──────────────────────",
                f"  Avg temp    : {prev7['TD_avg'].mean():.1f}°C  (max {prev7['TD_max'].max():.1f}°C  min {prev7['TD_min'].min():.1f}°C)" if 'TD_avg' in prev7.columns else "",
                f"  Total rain  : {prev7['Rain_total'].sum():.1f} mm" if 'Rain_total' in prev7.columns else "",
                f"  Avg humidity: {prev7['RH_avg'].mean():.0f}%" if 'RH_avg' in prev7.columns else "",
                f"  Frost days  : {frost7}",
            ]

        if len(prev30) > len(prev7) + 3:  # only add 30-day if it meaningfully extends the 7-day window
            n30 = len(prev30)
            frost30 = int((prev30['TD_min'] < 0).sum()) if 'TD_min' in prev30.columns else 'N/A'
            lines += [
                "",
                f"── PAST {n30} DAYS (up to {actual_date}) ──────────────────────",
                f"  Avg temp    : {prev30['TD_avg'].mean():.1f}°C  (max {prev30['TD_max'].max():.1f}°C  min {prev30['TD_min'].min():.1f}°C)" if 'TD_avg' in prev30.columns else "",
                f"  Total rain  : {prev30['Rain_total'].sum():.1f} mm" if 'Rain_total' in prev30.columns else "",
                f"  Avg humidity: {prev30['RH_avg'].mean():.0f}%" if 'RH_avg' in prev30.columns else "",
                f"  Frost days  : {frost30}",
            ]

        lines += [
            "",
            "Use the above historical context for long-term agricultural planning decisions.",
        ]

        return "\n".join(l for l in lines if l is not None)
