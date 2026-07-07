import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# =========================
# Configuración general
# =========================
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_PATH = "santa_teresa_serie_larga_5057.csv"
OUTPUT_DIR = "outputs_5057"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATE_COL = "fecha_hora"
UNIT_COL = "unidad"
TARGET_COL = "potencia_efectiva"

WINDOWS = [4, 12, 24]

# Partición cronológica usada en el experimento
TRAIN_END = pd.to_datetime("2025-02-23 21:30:00")
VAL_END = pd.to_datetime("2025-03-13 20:45:00")

# =========================
# Funciones auxiliares
# =========================
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def build_sequences_unit(df_unit, window):
    values = df_unit[TARGET_COL].values.reshape(-1, 1)
    dates = df_unit[DATE_COL].values
    units = df_unit[UNIT_COL].values

    X, y, y_dates, y_units = [], [], [], []

    for i in range(window, len(values)):
        X.append(values[i-window:i])
        y.append(values[i])
        y_dates.append(dates[i])
        y_units.append(units[i])

    return np.array(X), np.array(y).reshape(-1, 1), np.array(y_dates), np.array(y_units)

def create_lstm_model(window):
    model = Sequential([
        LSTM(64, input_shape=(window, 1)),
        Dropout(0.15),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def plot_predictions(pred_df, window):
    plt.figure(figsize=(12, 5))

    for unit in sorted(pred_df[UNIT_COL].unique()):
        temp = pred_df[pred_df[UNIT_COL] == unit].copy()
        plt.plot(temp[DATE_COL], temp["real"], label=f"Real {unit}", linewidth=1)
        plt.plot(temp[DATE_COL], temp["prediccion"], label=f"LSTM {unit}", linestyle="--", linewidth=1)

    plt.title(f"Predicción LSTM con ventana de {window} registros")
    plt.xlabel("Fecha")
    plt.ylabel("Potencia efectiva (MW)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, f"predicciones_lstm_g1_g2_w{window}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

# =========================
# Carga de datos
# =========================
df = pd.read_csv(DATA_PATH)
df[DATE_COL] = pd.to_datetime(df[DATE_COL])
df = df.sort_values([UNIT_COL, DATE_COL]).reset_index(drop=True)

all_preds = []
all_metrics = []

for window in WINDOWS:
    print(f"\nEntrenando LSTM con ventana {window}...")

    preds_window = []

    for unit in sorted(df[UNIT_COL].unique()):
        print(f"  Unidad {unit}")

        df_unit = df[df[UNIT_COL] == unit].copy().sort_values(DATE_COL)

        train_df = df_unit[df_unit[DATE_COL] <= TRAIN_END].copy()
        val_df = df_unit[(df_unit[DATE_COL] > TRAIN_END) & (df_unit[DATE_COL] <= VAL_END)].copy()
        test_df = df_unit[df_unit[DATE_COL] > VAL_END].copy()

        # Ajuste del escalador solo con entrenamiento
        scaler = MinMaxScaler()
        scaler.fit(train_df[[TARGET_COL]])

        df_unit_scaled = df_unit.copy()
        df_unit_scaled[TARGET_COL] = scaler.transform(df_unit[[TARGET_COL]])

        X, y, dates, units = build_sequences_unit(df_unit_scaled, window)

        dates_pd = pd.to_datetime(dates)

        train_mask = dates_pd <= TRAIN_END
        val_mask = (dates_pd > TRAIN_END) & (dates_pd <= VAL_END)
        test_mask = dates_pd > VAL_END

        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        dates_test = dates[test_mask]
        units_test = units[test_mask]

        model = create_lstm_model(window)

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True
        )

        start = time.perf_counter()
        model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=80,
            batch_size=32,
            callbacks=[early_stop],
            verbose=0
        )

        y_pred_scaled = model.predict(X_test, verbose=0)
        elapsed = time.perf_counter() - start

        y_test_inv = scaler.inverse_transform(y_test).ravel()
        y_pred_inv = scaler.inverse_transform(y_pred_scaled).ravel()

        pred = pd.DataFrame({
            DATE_COL: pd.to_datetime(dates_test),
            UNIT_COL: units_test,
            "real": y_test_inv,
            "prediccion": y_pred_inv,
            "modelo": "LSTM",
            "ventana": window
        })

        pred["error_absoluto"] = (pred["real"] - pred["prediccion"]).abs()
        preds_window.append(pred)

        all_metrics.append({
            "modelo": "LSTM",
            "ventana": window,
            "unidad": unit,
            "MAE": mean_absolute_error(y_test_inv, y_pred_inv),
            "RMSE": rmse(y_test_inv, y_pred_inv),
            "MAPE": mape(y_test_inv, y_pred_inv),
            "tiempo_segundos": elapsed
        })

    pred_window = pd.concat(preds_window, ignore_index=True)

    # Métrica agregada G1 + G2
    all_metrics.append({
        "modelo": "LSTM",
        "ventana": window,
        "unidad": "Agregado",
        "MAE": mean_absolute_error(pred_window["real"], pred_window["prediccion"]),
        "RMSE": rmse(pred_window["real"], pred_window["prediccion"]),
        "MAPE": mape(pred_window["real"], pred_window["prediccion"]),
        "tiempo_segundos": np.nan
    })

    plot_predictions(pred_window, window)
    all_preds.append(pred_window)

# Exportar predicciones LSTM corregidas
preds_lstm = pd.concat(all_preds, ignore_index=True)
metrics_lstm = pd.DataFrame(all_metrics)

preds_lstm.to_csv(
    os.path.join(OUTPUT_DIR, "predicciones_lstm_g1_g2_5057.csv"),
    index=False,
    encoding="utf-8-sig"
)

metrics_lstm.to_csv(
    os.path.join(OUTPUT_DIR, "metricas_lstm_g1_g2_5057.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("\nMétricas LSTM G1, G2 y agregado:")
print(metrics_lstm.to_string(index=False))
print("\nArchivos generados en outputs_5057.")