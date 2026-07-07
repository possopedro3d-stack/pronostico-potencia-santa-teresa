# -*- coding: utf-8 -*-
"""
Experimento de pronóstico - Central Santa Teresa, base desagregada por unidad generadora
Modelos: Baseline de persistencia, Random Forest, XGBoost y LSTM
Variable objetivo: potencia_efectiva por unidad generadora
Frecuencia: 15 minutos
"""
import os, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

DATA_FILE = "santa_teresa_serie_larga_5057.csv"
OUTPUT_DIR = "outputs_5057"
TARGET = "potencia_efectiva"
DATE_COL = "fecha_hora"
WINDOWS = [4, 12, 24]  # 1 hora, 3 horas, 6 horas
HORIZON = 1
os.makedirs(OUTPUT_DIR, exist_ok=True)

def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() else np.nan

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def build_supervised_panel(df, target_col, window, horizon=1):
    frames = []
    for unidad, g in df.groupby("unidad", sort=False):
        data = g.sort_values(DATE_COL).copy()
        for lag in range(1, window + 1):
            data[f"lag_{lag}"] = data[target_col].shift(lag)
        data["media_movil_4"] = data[target_col].shift(1).rolling(4).mean()
        data["media_movil_12"] = data[target_col].shift(1).rolling(12).mean()
        data["media_movil_24"] = data[target_col].shift(1).rolling(24).mean()
        data["y"] = data[target_col].shift(-horizon)
        frames.append(data)
    data = pd.concat(frames, ignore_index=True)
    data = data.dropna().sort_values([DATE_COL, "unidad"]).reset_index(drop=True)
    feature_cols = [f"lag_{i}" for i in range(1, window + 1)] + ["media_movil_4", "media_movil_12", "media_movil_24", "unidad_id", "hora", "minuto", "dia_semana", "mes"]
    return data, feature_cols

def chronological_split(n, train_ratio=0.70, val_ratio=0.15):
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return np.arange(0, train_end), np.arange(train_end, val_end), np.arange(val_end, n)

def build_lstm_panel_sequences(df, window, horizon=1):
    X, y, dates, units = [], [], [], []
    for unidad, g in df.groupby("unidad", sort=False):
        g = g.sort_values(DATE_COL).reset_index(drop=True)
        vals = g[TARGET].values.astype(float)
        unidad_id = g["unidad_id"].values.astype(float)
        hours = g["hora"].values.astype(float)
        minutes = g["minuto"].values.astype(float)
        dow = g["dia_semana"].values.astype(float)
        months = g["mes"].values.astype(float)
        # features per timestep: target history + unit/time context
        features = np.column_stack([vals, unidad_id, hours, minutes, dow, months])
        for i in range(window, len(g) - horizon + 1):
            X.append(features[i-window:i, :])
            y.append(vals[i + horizon - 1])
            dates.append(g.loc[i + horizon - 1, DATE_COL])
            units.append(unidad)
    return np.array(X), np.array(y), np.array(dates), np.array(units)

def save_prediction_plot(plot_df, model_name, window):
    plt.figure(figsize=(12, 5))
    for unidad in plot_df["unidad"].unique():
        sub = plot_df[plot_df["unidad"] == unidad]
        plt.plot(sub[DATE_COL], sub["real"], linewidth=1.6, label=f"Real {unidad}")
        plt.plot(sub[DATE_COL], sub["prediccion"], linewidth=1.6, linestyle="--", label=f"Predicción {unidad}")
    plt.title(f"Predicción vs valor real - {model_name} - ventana {window}")
    plt.xlabel("Fecha y hora")
    plt.ylabel(TARGET)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    safe = model_name.lower().replace(" ", "_")
    plt.savefig(os.path.join(OUTPUT_DIR, f"predicciones_{safe}_w{window}.png"), dpi=180)
    plt.close()

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"No se encontró {DATA_FILE}. Coloque este script en la misma carpeta que el CSV.")

df = pd.read_csv(DATA_FILE)
df[DATE_COL] = pd.to_datetime(df[DATE_COL])
df = df.sort_values(["unidad", DATE_COL]).reset_index(drop=True)
# Reforzar variables temporales por si el CSV se abrió/modificó
df["hora"] = df[DATE_COL].dt.hour
df["minuto"] = df[DATE_COL].dt.minute
df["dia_semana"] = df[DATE_COL].dt.dayofweek
df["mes"] = df[DATE_COL].dt.month
if "unidad_id" not in df.columns:
    df["unidad_id"] = df["unidad"].map({"G1": 1, "G2": 2}).astype(int)

print("Base desagregada cargada correctamente")
print(f"Registros: {len(df)}")
print(f"Unidades: {', '.join(df['unidad'].unique())}")
print(f"Periodo: {df[DATE_COL].min()} a {df[DATE_COL].max()}")
print("-" * 70)

all_metrics, all_predictions = [], []
for window in WINDOWS:
    print(f"Procesando ventana temporal: {window}")
    supervised, feature_cols = build_supervised_panel(df, TARGET, window, HORIZON)
    idx_train, idx_val, idx_test = chronological_split(len(supervised))
    train, val, test = supervised.iloc[idx_train], supervised.iloc[idx_val], supervised.iloc[idx_test]
    X_train, y_train = train[feature_cols], train["y"]
    X_test, y_test = test[feature_cols], test["y"]

    # Baseline por unidad: último valor de la misma unidad
    start = time.perf_counter()
    y_pred = test["lag_1"].values
    elapsed = time.perf_counter() - start
    all_metrics.append({"modelo":"Baseline persistencia","ventana":window,"MAE":mean_absolute_error(y_test,y_pred),"RMSE":rmse(y_test,y_pred),"MAPE":mape(y_test,y_pred),"tiempo_segundos":elapsed})
    pred = pd.DataFrame({DATE_COL:test[DATE_COL].values,"unidad":test["unidad"].values,"real":y_test.values,"prediccion":y_pred,"modelo":"Baseline persistencia","ventana":window})
    all_predictions.append(pred); save_prediction_plot(pred, "Baseline persistencia", window)

    # RF
    start=time.perf_counter()
    rf=RandomForestRegressor(n_estimators=300, random_state=42, min_samples_leaf=2, n_jobs=-1)
    rf.fit(X_train,y_train); y_pred=rf.predict(X_test); elapsed=time.perf_counter()-start
    all_metrics.append({"modelo":"Random Forest","ventana":window,"MAE":mean_absolute_error(y_test,y_pred),"RMSE":rmse(y_test,y_pred),"MAPE":mape(y_test,y_pred),"tiempo_segundos":elapsed})
    pred=pd.DataFrame({DATE_COL:test[DATE_COL].values,"unidad":test["unidad"].values,"real":y_test.values,"prediccion":y_pred,"modelo":"Random Forest","ventana":window})
    all_predictions.append(pred); save_prediction_plot(pred,"Random Forest",window)

    # XGB
    start=time.perf_counter()
    xgb=XGBRegressor(n_estimators=400, learning_rate=0.03, max_depth=4, subsample=0.9, colsample_bytree=0.9, objective="reg:squarederror", random_state=42)
    xgb.fit(X_train,y_train); y_pred=xgb.predict(X_test); elapsed=time.perf_counter()-start
    all_metrics.append({"modelo":"XGBoost","ventana":window,"MAE":mean_absolute_error(y_test,y_pred),"RMSE":rmse(y_test,y_pred),"MAPE":mape(y_test,y_pred),"tiempo_segundos":elapsed})
    pred=pd.DataFrame({DATE_COL:test[DATE_COL].values,"unidad":test["unidad"].values,"real":y_test.values,"prediccion":y_pred,"modelo":"XGBoost","ventana":window})
    all_predictions.append(pred); save_prediction_plot(pred,"XGBoost",window)

    # LSTM con secuencias por unidad
    X_seq,y_seq,dates,units=build_lstm_panel_sequences(df, window, HORIZON)
    idx_train_l, idx_val_l, idx_test_l=chronological_split(len(X_seq))
    X_train_l,y_train_l=X_seq[idx_train_l],y_seq[idx_train_l]
    X_val_l,y_val_l=X_seq[idx_val_l],y_seq[idx_val_l]
    X_test_l,y_test_l=X_seq[idx_test_l],y_seq[idx_test_l]
    dates_test, units_test=dates[idx_test_l], units[idx_test_l]

    sx=MinMaxScaler(); sy=MinMaxScaler()
    sx.fit(X_train_l.reshape(-1, X_train_l.shape[-1])); sy.fit(y_train_l.reshape(-1,1))
    X_train_s=sx.transform(X_train_l.reshape(-1, X_train_l.shape[-1])).reshape(X_train_l.shape)
    X_val_s=sx.transform(X_val_l.reshape(-1, X_val_l.shape[-1])).reshape(X_val_l.shape)
    X_test_s=sx.transform(X_test_l.reshape(-1, X_test_l.shape[-1])).reshape(X_test_l.shape)
    y_train_s=sy.transform(y_train_l.reshape(-1,1)).ravel(); y_val_s=sy.transform(y_val_l.reshape(-1,1)).ravel()

    start=time.perf_counter(); tf.keras.utils.set_random_seed(42)
    model=Sequential([LSTM(64, input_shape=(window, X_train_l.shape[-1])), Dropout(0.15), Dense(32, activation="relu"), Dense(1)])
    model.compile(optimizer="adam", loss="mse")
    early=EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    model.fit(X_train_s, y_train_s, validation_data=(X_val_s,y_val_s), epochs=80, batch_size=32, verbose=0, callbacks=[early])
    y_pred_s=model.predict(X_test_s, verbose=0).ravel(); y_pred=sy.inverse_transform(y_pred_s.reshape(-1,1)).ravel(); elapsed=time.perf_counter()-start
    all_metrics.append({"modelo":"LSTM","ventana":window,"MAE":mean_absolute_error(y_test_l,y_pred),"RMSE":rmse(y_test_l,y_pred),"MAPE":mape(y_test_l,y_pred),"tiempo_segundos":elapsed})
    pred=pd.DataFrame({DATE_COL:dates_test,"unidad":units_test,"real":y_test_l,"prediccion":y_pred,"modelo":"LSTM","ventana":window})
    all_predictions.append(pred); save_prediction_plot(pred,"LSTM",window)
    print(f"Ventana {window} terminada.")
    print("-" * 70)

metrics=pd.DataFrame(all_metrics).sort_values(["MAE","RMSE"]).reset_index(drop=True)
preds=pd.concat(all_predictions, ignore_index=True)
preds["error_absoluto"]=(preds["real"]-preds["prediccion"]).abs()
metrics.to_csv(os.path.join(OUTPUT_DIR,"metricas_modelos_5057.csv"), index=False, encoding="utf-8-sig")
preds.to_csv(os.path.join(OUTPUT_DIR,"predicciones_modelos_5057.csv"), index=False, encoding="utf-8-sig")
preds.sort_values("error_absoluto", ascending=False).head(20).to_csv(os.path.join(OUTPUT_DIR,"analisis_fallos_top20_5057.csv"), index=False, encoding="utf-8-sig")
print("\nExperimento 5057 finalizado correctamente.")
print("Resultados guardados en outputs_5057.")
print(metrics.to_string(index=False))
