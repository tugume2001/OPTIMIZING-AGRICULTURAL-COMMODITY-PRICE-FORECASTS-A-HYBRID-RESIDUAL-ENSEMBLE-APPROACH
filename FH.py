"""
Fair Harvest — Hybrid Residual Ensemble (v56.0 - "The Clean Report Edition")
- GRAPH FIX: Restored the exact Plotly White structure from Plantains_Owino_Forecast.html.
- HOLIDAY PRIME: Prophet model is now officially primed for 'UG' (Uganda).
- UNIFIED BASE: Fuses EXPERIMENTS.py resilience with the Lightning API.
- GUI: Premium Dark Mode aesthetic for the desktop app.
- AI CORE: 20-Second Parallel Ensemble (Prophet + XGBoost + LSTM).
"""

import os
import random
import numpy as np
import pandas as pd
import logging
import warnings
from datetime import datetime, timedelta
import re
from sklearn.metrics import mean_squared_error, mean_absolute_error
import queue
import time
import threading
import traceback
import concurrent.futures
import requests
import pickle
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import plotly.graph_objects as go
from plotly.offline import plot as plotly_plot

# Suppress TensorFlow/CPU warnings completely
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
tf.get_logger().setLevel('ERROR') 

# ---------------------------
# API CONFIGURATIONS
# ---------------------------
TELEGRAM_BOT_TOKEN = "7974415513:AAFLzp0h7y8F1efapwfV5xJPzOOjGQWDzQE" 
TELEGRAM_CHAT_ID = "-1003648321965" 
VISUAL_CROSSING_API_KEY = "U7M2JX5A9G2XS8RT7UTSRFAWF" # <--- INSERT YOUR KEY HERE

WFP_CSV_URL = "https://data.humdata.org/dataset/883929b1-521e-4834-97f5-0ccc2df75b89/resource/e082d683-cad5-4dcd-bf54-db76ae254d33/download/wfp_food_prices_uga.csv"

try:
    import telebot
except ImportError:
    telebot = None

from prophet import Prophet
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import HuberRegressor
from sklearn.ensemble import IsolationForest 

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor
    HAS_XGB = False

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K

# ---------------------------
# Data & External APIs
# ---------------------------

def get_desktop_folder():
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    folder = os.path.join(desktop, "COMMODITY_FORECASTS")
    if not os.path.exists(folder): os.makedirs(folder)
    return folder

def clean_filename(s):
    return re.sub(r'(?u)[^-\w.]', '', str(s).strip().replace(' ', '_'))

def fetch_weather_insight():
    if not VISUAL_CROSSING_API_KEY or VISUAL_CROSSING_API_KEY == "YOUR_VISUAL_CROSSING_API_KEY":
        return "🌦️ <i>Weather Context: Disabled (Visual Crossing API Key Required)</i>"
    try:
        url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/Kampala,Uganda?unitGroup=metric&key={VISUAL_CROSSING_API_KEY}&contentType=json"
        resp = requests.get(url, timeout=5)
        if resp.status_code != 200:
            return "🌦️ <i>Weather Context: Service currently unreachable.</i>"
            
        data = resp.json()
        precip = sum([day.get('precip', 0) for day in data['days'][:14]])
        
        if precip > 50:
            return f"🌧️ <b>Weather Alert:</b> Heavy rainfall expected ({precip:.1f}mm). Watch for muddy roads causing transport disruptions."
        elif precip < 5:
            return f"☀️ <b>Weather Alert:</b> Very dry conditions expected ({precip:.1f}mm). Favorable for transport, but prolonged drought may impact yields."
        else:
            return f"⛅ <b>Weather Alert:</b> Normal conditions expected ({precip:.1f}mm). Supply chains should operate normally."
    except Exception as e:
        return f"🌦️ <i>Weather Context: Data error ({str(e)})</i>"

def get_market_sentiment(series):
    if len(series) < 90: return "Neutral (Insufficient Data)"
    short_term = series.iloc[-30:].mean()
    long_term = series.iloc[-90:].mean()
    
    if short_term > long_term * 1.05:
        return "Bullish 📈 (Recent prices are surging above the 90-day average)"
    elif short_term < long_term * 0.95:
        return "Bearish 📉 (Recent prices are dropping below the 90-day average)"
    else:
        return "Neutral ⚖️ (Prices are stable compared to the 90-day baseline)"

# ---------------------------
# Core ML Logic (PARALLEL ENSEMBLE)
# ---------------------------

def train_test_split_ts(series, test_ratio=0.15):
    n = len(series)
    n_test = max(1, int(n * test_ratio))
    return series.iloc[:-n_test], series.iloc[-n_test:]

def prepare_series_data(df, date_col, value_col, freq_code):
    df = df.copy()
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col, value_col]).sort_values(date_col).set_index(date_col)
    
    df_res = df[[value_col]].resample(freq_code).mean()
    df_interp = df_res.copy().interpolate(method='time').ffill().bfill()
    return {'interp': (df_interp[value_col], df_interp[value_col].ewm(span=12).mean())}

def fit_linear_trend(series, steps_ahead):
    X = np.arange(len(series)).reshape(-1, 1)
    model = HuberRegressor().fit(X, series.values)
    return model.predict(X), model.predict(np.arange(len(series), len(series) + steps_ahead).reshape(-1, 1))

def prophet_forecast(fit_series, test_steps, fut_steps, p_scale, country, is_monthly):
    df_fit = pd.DataFrame({'ds': fit_series.index, 'y': fit_series.values})
    m = Prophet(changepoint_prior_scale=p_scale, uncertainty_samples=0)
    
    # Uganda Holiday Prime applied here
    if country != 'None':
        m.add_country_holidays(country_name=country)
        
    m.fit(df_fit)
    fut = m.make_future_dataframe(periods=test_steps + fut_steps, freq='MS' if is_monthly else 'D', include_history=False)
    fc = m.predict(fut)
    return fc['yhat'].values[:test_steps], fc['yhat'].values[test_steps:]

def create_hybrid_features(data_series, lags=7):
    df = pd.DataFrame(data_series.values, columns=['val'], index=data_series.index)
    df['month'] = df.index.month
    for l in range(1, lags + 1): df[f'lag_{l}'] = df['val'].shift(l)
    df = df.dropna()
    if len(df) == 0: return np.array([]), np.array([])
    return df.drop(['val', 'month'], axis=1).values, df['val'].values

def xgb_forecast(fit_residuals_series, scaler, test_steps, fut_steps, lags, n_est, learn_rate):
    vals = fit_residuals_series.values.reshape(-1, 1)
    scaled = scaler.transform(vals).flatten()
    X, y = create_hybrid_features(pd.Series(scaled, index=fit_residuals_series.index), lags)
    if len(X) < 5: return None, None
    model = XGBRegressor(n_estimators=n_est, learning_rate=learn_rate, max_depth=4) if HAS_XGB else GradientBoostingRegressor(n_estimators=n_est, learning_rate=learn_rate, max_depth=4)
    model.fit(X, y)

    combined_steps = test_steps + fut_steps
    preds, curr_seq = [], list(scaled)
    for i in range(combined_steps):
        feat = np.array(curr_seq[-lags:]).reshape(1, -1)
        p = model.predict(feat)[0]
        preds.append(p); curr_seq.append(p)
    inv = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    return inv[:test_steps], inv[test_steps:]

def lstm_forecast(fit_residuals, scaler, test_steps, fut_steps, units, window):
    scaled = scaler.transform(fit_residuals.reshape(-1, 1))
    X, y = [], []
    for i in range(window, len(scaled)):
        X.append(scaled[i-window:i, 0]); y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    if len(X) < 5: return None, None
    K.clear_session()
    
    model = Sequential([
        Input(shape=(window, 1)),
        LSTM(units, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    
    early_stop = EarlyStopping(monitor='loss', patience=4, restore_best_weights=True)
    model.fit(X.reshape(X.shape[0],X.shape[1],1), y, epochs=40, batch_size=16, verbose=0, callbacks=[early_stop])
    
    curr = scaled[-window:].reshape(1, window, 1)
    preds = []
    for _ in range(test_steps + fut_steps):
        p = model.predict(curr, verbose=0)[0, 0]
        preds.append(p)
        curr = np.concatenate([curr[:, 1:, :], [[[p]]]], axis=1)
    inv = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    return inv[:test_steps], inv[test_steps:]

def run_fast_ensemble(data_variants, steps, is_monthly, country, use_rw):
    fit_main, test_main = train_test_split_ts(data_variants['interp'][1], test_ratio=0.15)
    test_steps = len(test_main)
    if test_steps < 2: return None
    
    trend_fit, trend_future = fit_linear_trend(fit_main, test_steps + steps)
    residuals_fit = fit_main - pd.Series(trend_fit, index=fit_main.index)
    scaler = RobustScaler().fit(residuals_fit.values.reshape(-1, 1))
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        f_prophet = executor.submit(prophet_forecast, fit_main, test_steps, steps, 0.1, country, is_monthly)
        f_xgb = executor.submit(xgb_forecast, residuals_fit, scaler, test_steps, steps, 7, 100, 0.1)
        f_lstm = executor.submit(lstm_forecast, residuals_fit.values, scaler, test_steps, steps, 32, 7)

        tp_p, fp_p = f_prophet.result()
        tp_x_raw, fp_x_raw = f_xgb.result()
        tp_l_raw, fp_l_raw = f_lstm.result()

    test_preds, fut_preds = [], []
    if tp_p is not None: test_preds.append(tp_p); fut_preds.append(fp_p)
    if tp_x_raw is not None: test_preds.append(tp_x_raw + trend_future[:test_steps]); fut_preds.append(fp_x_raw + trend_future[test_steps:])
    if tp_l_raw is not None: test_preds.append(tp_l_raw + trend_future[:test_steps]); fut_preds.append(fp_l_raw + trend_future[test_steps:])

    if not test_preds: return None

    ens_test = np.mean(test_preds, axis=0)
    ens_fut = np.mean(fut_preds, axis=0)
    
    mape = np.mean(np.abs((test_main.values - ens_test) / test_main.values)) * 100
    mse = mean_squared_error(test_main.values, ens_test)
    std_err = np.std(test_main.values - ens_test)
    
    return {
        'accuracy': max(0, 100 - mape), 'mape': mape, 'mse': mse, 
        'ens_test': ens_test, 'ens_fut': ens_fut,
        'fut_high': ens_fut + (1.96 * std_err * np.sqrt(np.arange(1, steps + 1))),
        'fut_low': np.maximum(0, ens_fut - (1.96 * std_err * np.sqrt(np.arange(1, steps + 1)))),
        'test_dates': test_main.index, 'fit_series': fit_main
    }

# ---------------------------
# Main App Class
# ---------------------------

class ForecastApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fair Harvest Hybrid (v56.0) - The Clean Report Edition")
        self.root.geometry("1200x850")
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.df_full = None
        self.gui_queue = queue.Queue()
        self.task_queue = queue.Queue()
        self.output_dir = get_desktop_folder()
        
        self.cache_file = os.path.join(self.output_dir, "forecast_cache.pkl")
        self.forecast_cache = self._load_disk_cache()
        
        self.is_processing = False
        self.is_fetching = False
        self.boot_message_sent = False
        self.last_export_df = None  
        self.last_html_path = None
        
        self.bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN) if telebot else None
        self._build_ui()
        self.root.after(100, self._process_gui_queue)
        self.root.after(1000, self._process_task_queue)
        
        if self.bot: self._start_bot()

    def _load_disk_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                return {}
        return {}

    def _save_disk_cache(self):
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.forecast_cache, f)
        except Exception as e:
            self.log(f"Cache save warning: {e}")

    def _build_ui(self):
        # Desktop GUI stays Dark and Sleek
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        f1 = ctk.CTkFrame(self.root, fg_color="#1e1e1e")
        f1.pack(fill='x', padx=10, pady=10)
        
        self.btn_fetch = ctk.CTkButton(f1, text="🌐 AUTO-FETCH WFP DATA", font=("Arial", 14, "bold"), 
                                       fg_color="#0288D1", hover_color="#01579B", command=self.fetch_wfp_data)
        self.btn_fetch.pack(side='left', padx=10, pady=10)
        
        self.lbl_status = ctk.CTkLabel(f1, text="Status: Waiting for Data", text_color="#FF5252", font=("Arial", 12, "bold"))
        self.lbl_status.pack(side='left', padx=20)

        f2 = ctk.CTkFrame(self.root, fg_color="#1e1e1e")
        f2.pack(fill='x', padx=10, pady=5)
        
        self.var_date = tk.StringVar(value="date")
        self.var_price = tk.StringVar(value="price")
        
        ctk.CTkLabel(f2, text="Market:", font=("Arial", 12, "bold")).pack(side='left', padx=5)
        self.var_m_col = tk.StringVar(value="market")
        self.var_m_val = tk.StringVar(value="All")
        self.opt_m_val = ctk.CTkOptionMenu(f2, variable=self.var_m_val, values=["All"], fg_color="#333333", button_color="#444444")
        self.opt_m_val.pack(side='left', padx=5)

        ctk.CTkLabel(f2, text="Commodity:", font=("Arial", 12, "bold")).pack(side='left', padx=5)
        self.var_c_col = tk.StringVar(value="commodity")
        self.var_c_val = tk.StringVar(value="All")
        self.opt_c_val = ctk.CTkOptionMenu(f2, variable=self.var_c_val, values=["All"], fg_color="#333333", button_color="#444444")
        self.opt_c_val.pack(side='left', padx=5)

        ctk.CTkLabel(f2, text="Price Type:", font=("Arial", 12, "bold")).pack(side='left', padx=5)
        self.var_t_col = tk.StringVar(value="pricetype")
        self.var_t_val = tk.StringVar(value="All")
        self.opt_t_val = ctk.CTkOptionMenu(f2, variable=self.var_t_val, values=["All"], fg_color="#333333", button_color="#444444")
        self.opt_t_val.pack(side='left', padx=5)

        f4 = ctk.CTkFrame(self.root, fg_color="#1e1e1e")
        f4.pack(fill='x', padx=10, pady=10)
        
        ctk.CTkLabel(f4, text="Days to Predict:").pack(side='left', padx=5)
        self.var_steps = tk.StringVar(value="30")
        ctk.CTkEntry(f4, textvariable=self.var_steps, width=60, fg_color="#333333").pack(side='left', padx=5)
        
        self.var_force = ctk.BooleanVar(value=False)
        self.chk_force = ctk.CTkCheckBox(f4, text="Force Retrain", variable=self.var_force, text_color="#FF5252")
        self.chk_force.pack(side='left', padx=20)
        
        self.btn_run = ctk.CTkButton(f4, text="▶ RUN LOCAL FORECAST", font=("Arial", 12, "bold"), fg_color="#00C853", hover_color="#00E676", text_color="black", command=self.start_local)
        self.btn_run.pack(side='left', padx=20)
        
        self.btn_export = ctk.CTkButton(f4, text="💾 EXPORT EXCEL", font=("Arial", 12, "bold"), fg_color="#6A1B9A", hover_color="#8E24AA", command=self.export_to_excel)
        self.btn_export.pack(side='left', padx=10)

        self.btn_save_graph = ctk.CTkButton(f4, text="📊 SAVE GRAPH", font=("Arial", 12, "bold"), fg_color="#E64A19", hover_color="#D84315", command=self.save_graph)
        self.btn_save_graph.pack(side='left', padx=10)
        
        self.prog = ctk.CTkProgressBar(f4, width=150, progress_color="#00C853"); self.prog.set(0); self.prog.pack(side='left', padx=20)

        self.txt_log = ctk.CTkTextbox(self.root, height=350, font=("Consolas", 12), fg_color="#0d0d0d", text_color="#00E676")
        self.txt_log.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.log(f"System initialized. {len(self.forecast_cache)} predictions loaded from disk cache.")

    def log(self, m): 
        self.txt_log.insert('end', f"[{datetime.now().strftime('%H:%M:%S')}] {m}\n")
        self.txt_log.see('end')
        
    def on_closing(self):
        self.log("Saving cache and initiating Safe Shutdown Sequence...")
        self._save_disk_cache()
        if self.bot and self.boot_message_sent:
            try:
                self.bot.send_message(TELEGRAM_CHAT_ID, "🔴 <b>FAIR HARVEST SYSTEM OFFLINE</b>\n\nThe server is shutting down. Market predictions are temporarily paused.", parse_mode="HTML")
            except Exception: pass
        self.root.destroy()
        os._exit(0)

    def fetch_wfp_data(self):
        if self.is_fetching: return
        self.is_fetching = True
        self.btn_fetch.configure(state="disabled")
        self.lbl_status.configure(text="Status: Downloading Live WFP Data...", text_color="#FFB300")
        self.log("Connecting to UN Humanitarian Data Exchange (WFP)...")
        threading.Thread(target=self._download_and_convert, daemon=True).start()

    def _download_and_convert(self):
        try:
            response = requests.get(WFP_CSV_URL)
            response.raise_for_status()
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            if df.iloc[0].astype(str).str.startswith('#').any():
                df = df.iloc[1:].reset_index(drop=True)
            excel_path = os.path.join(self.output_dir, "WFP_Uganda_Live_Data.xlsx")
            df.to_excel(excel_path, index=False)
            self.gui_queue.put(("wfp_loaded", (df, excel_path)))
        except Exception as e:
            self.gui_queue.put(("log", f"ERROR DOWNLOADING DATA: {str(e)}"))
            self.gui_queue.put(("wfp_fail", None))

    def save_graph(self):
        if not self.last_html_path: 
            messagebox.showinfo("Info", "No graph generated yet. Run a forecast first.")
            return
        try:
            f = filedialog.asksaveasfilename(defaultextension=".html", filetypes=[("HTML", "*.html")])
            if f:
                shutil.copy(self.last_html_path, f)
                self.log(f"Graph copy saved manually.")
                messagebox.showinfo("Success", f"Dashboard HTML exported successfully to:\n{f}")
        except Exception as e: 
            self.log(f"Save graph error: {e}")

    def export_to_excel(self):
        if self.last_export_df is None:
            messagebox.showinfo("Export Error", "Run a forecast first before exporting data.")
            return
        try:
            fname = clean_filename(f"{self.var_c_val.get()}_{self.var_m_val.get()}_Data")
            dest = os.path.join(self.output_dir, f"{fname}.xlsx")
            attempts = 0
            while attempts < 5:
                try:
                    self.last_export_df.to_excel(dest, index=False)
                    break 
                except PermissionError:
                    dest = os.path.join(self.output_dir, f"{fname}_{int(time.time())}.xlsx")
                    attempts += 1
            if attempts >= 5:
                messagebox.showerror("Error", "Could not save file. Please close Excel and try again.")
                return
            messagebox.showinfo("Success", f"Data exported to:\n{dest}")
            self.log(f"Exported Excel to: {os.path.basename(dest)}")
        except Exception as e:
            self.log(f"Excel Export Failed: {e}")

    def start_local(self):
        if self.df_full is None: return messagebox.showerror("Error", "Please fetch data first.")
        cfg = {
            'source': 'local', 'user': 'Admin', 'steps': int(self.var_steps.get()),
            'l1_col': self.var_c_col.get(), 'l1_val': self.var_c_val.get(),
            'l2_col': self.var_m_col.get(), 'l2_val': self.var_m_val.get(),
            'l3_col': self.var_t_col.get(), 'l3_val': self.var_t_val.get(),
            'date': self.var_date.get(), 'price': self.var_price.get(), 
            'monthly': False, 'country': 'None', 'rw': True,
            'force_retrain': self.var_force.get(),
            'telegram_msg_id': None
        }
        self.task_queue.put(cfg)

    def _start_bot(self):
        @self.bot.message_handler(commands=['start', 'help'])
        def send_welcome(m):
            if self.df_full is None:
                self.bot.reply_to(m, "Welcome to Fair Harvest Bot! 🌾\n\nI am pulling the latest Live WFP Market Data. This will take ~10 seconds...")
                self.fetch_wfp_data()
            else:
                self.bot.reply_to(m, "Welcome to Fair Harvest! 🌾\n\nUse /menu to see available data.\nFormat: /predict [market] [commodity] [type] [days]")

        @self.bot.message_handler(commands=['menu'])
        def show_menu(m):
            if self.df_full is None: return self.bot.reply_to(m, "Data is currently downloading. Try again in a few seconds.")
            try:
                menu_text = "<b>Available Data:</b>\n"
                menu_text += f"- Markets: {', '.join(self.df_full['market'].dropna().unique()[:5])}...\n"
                menu_text += f"- Commodities: {', '.join(self.df_full['commodity'].dropna().unique()[:5])}...\n"
                self.bot.reply_to(m, menu_text, parse_mode='HTML')
            except Exception as e: self.bot.reply_to(m, f"Menu Error: {str(e)}")

        @self.bot.message_handler(commands=['predict'])
        def h_pred(m):
            if self.df_full is None: return self.bot.reply_to(m, "System is initializing. Please send /start to load the database.")
            p = m.text.split()
            if len(p) < 5: return self.bot.reply_to(m, "Format: /predict [market] [commodity] [type] [days]")
            
            status_msg = self.bot.reply_to(m, f"⚡ <b>Initializing AI Ensemble</b>\n\nRunning Prophet, XGBoost, and LSTM concurrently for {p[2]} at {p[1]}...\n<i>Estimating completion in under 60 seconds.</i>", parse_mode="HTML")
            cfg = {
                'source': 'telegram', 'chat_id': m.chat.id, 'user': m.from_user.first_name,
                'telegram_msg_id': status_msg.message_id, 
                'l1_col': self.var_c_col.get(), 'l1_val': p[2], 'l2_col': self.var_m_col.get(), 'l2_val': p[1],
                'l3_col': self.var_t_col.get(), 'l3_val': p[3], 'steps': int(p[4]),
                'date': self.var_date.get(), 'price': self.var_price.get(), 
                'monthly': False, 'country': 'None', 'rw': True,
                'force_retrain': False
            }
            self.task_queue.put(cfg)
        threading.Thread(target=lambda: self.bot.polling(none_stop=True), daemon=True).start()

    def _process_gui_queue(self):
        while not self.gui_queue.empty():
            t, d = self.gui_queue.get()
            if t == "log": self.log(d)
            elif t == "prog": self.prog.set(d)
            elif t == "wfp_fail":
                self.btn_fetch.configure(state="normal")
                self.lbl_status.configure(text="Status: Download Failed", text_color="#FF5252")
                self.is_fetching = False
            elif t == "wfp_loaded":
                df, path = d
                self.df_full = df
                self.lbl_status.configure(text="Status: Live Data Loaded Active", text_color="#00C853")
                self.btn_fetch.configure(state="normal", text="🔄 UPDATE WFP DATA")
                self.log(f"Data successfully loaded -> {path}")
                self.is_fetching = False
                
                if self.bot and not self.boot_message_sent:
                    try:
                        self.bot.send_message(TELEGRAM_CHAT_ID, "🟢 <b>FAIR HARVEST SYSTEM ONLINE</b>\n\nThe AI ensemble (Prophet + XGBoost + LSTM) is synced and ready for lightning market analysis in Uganda.", parse_mode="HTML")
                        self.boot_message_sent = True
                    except Exception: pass

                try:
                    self.opt_m_val.configure(values=["All"] + list(df['market'].dropna().astype(str).unique()))
                    self.opt_c_val.configure(values=["All"] + list(df['commodity'].dropna().astype(str).unique()))
                    self.opt_t_val.configure(values=["All"] + list(df['pricetype'].dropna().astype(str).unique()))
                except KeyError: pass
        self.root.after(100, self._process_gui_queue)

    def _process_task_queue(self):
        if not self.is_processing and not self.task_queue.empty():
            cfg = self.task_queue.get()
            self.is_processing = True
            threading.Thread(target=self._run_task, args=(cfg,), daemon=True).start()
        self.root.after(1000, self._process_task_queue)

    def _run_task(self, cfg):
        req_desc = f"{cfg['l2_val']} {cfg['l1_val']} {cfg['l3_val']} ({cfg['steps']} days)"
        self.gui_queue.put(("log", f"\n--- LEDGER ENTRY: {cfg['user']} requested {req_desc} ---"))
        
        ckey = f"{cfg['l1_val']}_{cfg['l2_val']}_{cfg['l3_val']}_{cfg['steps']}".lower()
        telegram_msg_id = cfg.get('telegram_msg_id')
        
        if cfg['force_retrain']:
            self.gui_queue.put(("log", "Admin forced retrain. Bypassing cache..."))
        elif ckey in self.forecast_cache:
            self.gui_queue.put(("log", f"CACHE HIT for {ckey}. Sending instantly."))
            if telegram_msg_id and self.bot:
                try: self.bot.delete_message(cfg['chat_id'], telegram_msg_id)
                except: pass
            self._handle_res(cfg, self.forecast_cache[ckey], cached=True)
            self.is_processing = False
            return

        try:
            df = self.df_full.copy()
            for k, v in [('l1_col','l1_val'), ('l2_col','l2_val'), ('l3_col','l3_val')]:
                if cfg[k] != "None" and cfg[v] != "All":
                    df = df[df[cfg[k]].astype(str).str.contains(cfg[v], case=False, na=False)]
            
            if df.empty: raise ValueError("No data found for those exact parameters.")
                
            data_variants = prepare_series_data(df, cfg['date'], cfg['price'], 'D')
            self.gui_queue.put(("prog", 0.5))
            
            start_time = time.time()
            
            # NOTE: UG (Uganda) parameter is specifically passed here so Prophet predicts local holidays!
            res = run_fast_ensemble(data_variants, cfg['steps'], False, 'UG', True)
            
            elapsed = time.time() - start_time
            self.gui_queue.put(("log", f"Parallel Ensemble Math complete in {elapsed:.1f} seconds!"))
            
            if not res: raise ValueError("Model failed to converge on the data provided.")
                
            weather_text = fetch_weather_insight()
            date_series = pd.to_datetime(df[cfg['date']])
            
            payload = {
                'res': res, 
                'full_history': data_variants['interp'][0], 
                'dates': f"{date_series.min().date()} to {date_series.max().date()}",
                'weather': weather_text
            }
            
            self.forecast_cache[ckey] = payload
            self._save_disk_cache() 
            
            if telegram_msg_id and self.bot:
                try: self.bot.delete_message(cfg['chat_id'], telegram_msg_id)
                except: pass
                
            self._handle_res(cfg, payload, cached=False)
            
        except Exception as e: 
            err_msg = str(e)
            self.gui_queue.put(("log", f"ERROR: {err_msg}"))
            if cfg['source'] == 'telegram' and telegram_msg_id and self.bot:
                try: self.bot.edit_message_text(chat_id=cfg['chat_id'], message_id=telegram_msg_id, text=f"❌ Analysis failed: {err_msg}")
                except: pass
        finally: 
            self.is_processing = False
            self.gui_queue.put(("prog", 1.0))

    def _handle_res(self, cfg, payload, cached=False):
        res = payload['res']; hist = payload['full_history']
        
        # --- GRAPH GENERATION (Exactly matching Plantains_Owino_Forecast.html) ---
        test_dates = res['test_dates']
        fut_dates = [test_dates[-1] + timedelta(days=i) for i in range(1, cfg['steps'] + 1)]
        
        fig = go.Figure()
        
        # 1. Historical Data (Clean Dark Slate Blue Line, NO NEON FILL)
        fig.add_trace(go.Scatter(
            x=hist.index[-150:], y=hist.values[-150:], 
            name='Historical Trend', 
            line=dict(color='#2a3f5f', width=2),
            mode='lines',
            hovertemplate='<b>Date</b>: %{x}<br><b>Price</b>: %{y:,.0f} UGX<extra></extra>'
        ))
        
        # 2. Model Validation (Orange Dotted Line)
        fig.add_trace(go.Scatter(
            x=test_dates, y=res['ens_test'], 
            name='Model Validation', 
            line=dict(color='#FFA500', dash='dot', width=2),
            mode='lines'
        ))
        
        # 3. Future Prediction (Solid Red Line)
        f_x = [test_dates[-1]] + list(fut_dates)
        f_y = [hist.values[-1]] + list(res['ens_fut'])
        fig.add_trace(go.Scatter(
            x=f_x, y=f_y, 
            name='FUTURE PREDICTION', 
            line=dict(color='#FF3333', width=3),
            mode='lines',
            hovertemplate='<b>Target Date</b>: %{x}<br><b>Predicted</b>: %{y:,.0f} UGX<extra></extra>'
        ))
        
        # 4. Confidence Interval (Light Red Shading)
        fig.add_trace(go.Scatter(
            x=fut_dates, y=res['fut_high'], 
            fill=None, mode='lines', 
            line_color='rgba(0,0,0,0)', 
            showlegend=False, hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=fut_dates, y=res['fut_low'], 
            fill='tonexty', mode='lines', 
            line_color='rgba(0,0,0,0)', 
            fillcolor='rgba(255, 51, 51, 0.15)', 
            name='Risk Zone (95%)'
        ))

        # Premium Clean Layout (Plantains Replica)
        fig.update_layout(
            title=f"<b>Fair Harvest Market Intelligence: {cfg['l2_val']} {cfg['l1_val']}</b>",
            template='plotly_white',
            plot_bgcolor='white', 
            paper_bgcolor='white',
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(
                title="Timeline", 
                gridcolor='#DFE8F3', 
                zerolinecolor='#EBF0F8',
                zerolinewidth=2
            ),
            yaxis=dict(
                title="Price Output", 
                gridcolor='#DFE8F3', 
                zerolinecolor='#EBF0F8',
                zerolinewidth=2
            )
        )
        
        report_name = f"Report_{cfg['l1_val']}_{cfg['l2_val']}_{datetime.now().strftime('%H%M%S')}.html"
        self.last_html_path = os.path.join(self.output_dir, report_name)
        plotly_plot(fig, filename=self.last_html_path, auto_open=False)
        self.gui_queue.put(("log", f"SUCCESS: HTML Dashboard saved internally as {report_name}"))

        # --- PREPARE DATA FOR EXCEL EXPORT ---
        export_df = pd.DataFrame({
            'Date': fut_dates,
            'Forecasted_Price': res['ens_fut'],
            'Lowest_Estimate': res['fut_low'],
            'Highest_Estimate': res['fut_high']
        })
        self.last_export_df = export_df 

        # --- TELEGRAM TEXT REPORT GENERATION ---
        current_price = hist.values[-1]
        forecast_price = res['ens_fut'][-1]
        lowest_estimate = res['fut_low'][-1]
        highest_estimate = res['fut_high'][-1]
        accuracy = res['accuracy']
        
        percent_change = ((forecast_price - current_price) / current_price) * 100
        market_sentiment = get_market_sentiment(hist)
        
        if percent_change > 2.0:
            trend_emoji = "↗️"; trend_text = "Increasing Trend"
            outlook = "Prices are expected to rise. Consider securing inventory early."
        elif percent_change < -2.0:
            trend_emoji = "↘️"; trend_text = "Decreasing Trend"
            outlook = "Prices are expected to drop. Consider selling current stock soon."
        else:
            trend_emoji = "➡️"; trend_text = "Stable Market"
            outlook = "Prices are expected to remain relatively stable."
            
        vol_margin = highest_estimate - lowest_estimate
        if vol_margin > (current_price * 0.20): vol_risk = "High (Wide variance expected)"
        elif vol_margin > (current_price * 0.10): vol_risk = "Moderate"
        else: vol_risk = "Low (Highly predictable)"

        if cfg['source'] == 'telegram':
            msg = f"🌾 <b>FAIR HARVEST MARKET REPORT</b> 🌾\n"
            if cached: msg += "<i>(Retrieved from Secure Disk Cache)</i>\n"
            msg += f"<b>Market:</b> {cfg['l2_val']} (Uganda)\n"
            msg += f"<b>Commodity:</b> {cfg['l1_val']} ({cfg['l3_val']})\n"
            msg += f"<b>Target:</b> {cfg['steps']} Days Out\n\n"
            
            msg += f"📉 <b>MARKET SENTIMENT</b>\n"
            msg += f"• <b>Momentum:</b> {market_sentiment}\n\n"
            
            msg += f"📈 <b>PRICE PROJECTION</b>\n"
            msg += f"• <b>Current Price:</b> {current_price:,.0f} UGX\n"
            msg += f"• <b>Forecasted Price:</b> {forecast_price:,.0f} UGX\n"
            msg += f"• <b>Movement:</b> {trend_emoji} {percent_change:+.1f}% ({trend_text})\n\n"
            
            msg += f"📊 <b>RISK & VOLATILITY</b>\n"
            msg += f"• <b>Lowest Estimate:</b> {lowest_estimate:,.0f} UGX\n"
            msg += f"• <b>Highest Estimate:</b> {highest_estimate:,.0f} UGX\n"
            msg += f"• <b>Volatility:</b> {vol_risk}\n\n"
            
            msg += f"{payload.get('weather', '🌦️ <i>Weather Context: Historical cache (No weather data).</i>')}\n\n"
            
            msg += f"🤖 <b>MODEL TRANSPARENCY</b>\n"
            msg += f"• <b>Validation Accuracy:</b> {accuracy:.1f}%\n"
            msg += f"• <b>MAPE:</b> {res['mape']:.2f}% | <b>MSE:</b> {res['mse']:,.0f}\n"
            msg += f"• <b>Stack:</b> Prophet + XGBoost + LSTM\n"
            
            msg += f"💡 <b>OUTLOOK:</b> <i>{outlook}</i>"
            
            self.bot.send_message(cfg['chat_id'], msg, parse_mode='HTML')
            self.gui_queue.put(("log", f"SUCCESS: Sent Detailed Briefing to {cfg['user']}."))
            return

        if cfg['source'] == 'local':
            self.gui_queue.put(("log", f"SUCCESS: Local prediction completed. Accuracy: {accuracy:.1f}%"))

if __name__ == "__main__":
    root = ctk.CTk(); app = ForecastApp(root); root.mainloop()