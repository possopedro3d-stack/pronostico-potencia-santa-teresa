import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

OUTPUT_DIR = "outputs_5057"

OLD_PREDS = os.path.join(OUTPUT_DIR, "predicciones_modelos_5057.csv")
NEW_LSTM_PREDS = os.path.join(OUTPUT_DIR, "predicciones_lstm_g1_g2_5057.csv")
OLD_METRICS = os.path.join(OUTPUT_DIR, "metricas_modelos_5057.csv")
NEW_LSTM_METRICS = os.path.join(OUTPUT_DIR, "metricas_lstm_g1_g2_5057.csv")

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def format_number(x, decimals=2):
    if pd.isna(x):
        return "—"
    if abs(x) < 0.01 and x != 0:
        return "< 0,01"
    return f"{x:.{decimals}f}".replace(".", ",")

# =========================
# Cargar predicciones
# =========================
preds_old = pd.read_csv(OLD_PREDS)
preds_lstm_new = pd.read_csv(NEW_LSTM_PREDS)

# Se elimina el LSTM anterior porque solo tenía G2
preds_no_lstm = preds_old[preds_old["modelo"] != "LSTM"].copy()

# Se incorpora el LSTM corregido con G1 y G2
preds = pd.concat([preds_no_lstm, preds_lstm_new], ignore_index=True)

# Normalización de nombres
preds["modelo"] = preds["modelo"].replace({
    "Baseline persistencia": "Baseline persistencia",
    "Random Forest": "Random Forest",
    "XGBoost": "XGBoost",
    "LSTM": "LSTM"
})

# =========================
# Cargar tiempos de cómputo
# =========================
metrics_old = pd.read_csv(OLD_METRICS)
metrics_lstm = pd.read_csv(NEW_LSTM_METRICS)

time_lookup = {}

# Tiempos de modelos antiguos
for _, row in metrics_old.iterrows():
    model = row["modelo"]
    window = int(row["ventana"])
    if model != "LSTM":
        time_lookup[(model, window)] = row["tiempo_segundos"]

# Tiempos LSTM corregidos: suma G1 + G2 para el agregado
for window in sorted(metrics_lstm["ventana"].unique()):
    temp = metrics_lstm[(metrics_lstm["ventana"] == window) & (metrics_lstm["unidad"].isin(["G1", "G2"]))]
    time_lookup[("LSTM", int(window))] = temp["tiempo_segundos"].sum()

# =========================
# Calcular métricas desacopladas
# =========================
rows = []

model_order = ["Baseline persistencia", "Random Forest", "XGBoost", "LSTM"]
unit_order = ["G1", "G2", "Agregado"]

for window in [4, 12, 24]:
    for model in model_order:
        df_model = preds[(preds["modelo"] == model) & (preds["ventana"] == window)].copy()

        for unit in unit_order:
            if unit == "Agregado":
                df_eval = df_model.copy()
                tiempo = time_lookup.get((model, window), np.nan)
            else:
                df_eval = df_model[df_model["unidad"] == unit].copy()
                tiempo = np.nan

            if len(df_eval) == 0:
                continue

            rows.append({
                "modelo": model,
                "ventana": window,
                "unidad": unit,
                "MAE": mean_absolute_error(df_eval["real"], df_eval["prediccion"]),
                "RMSE": rmse(df_eval["real"], df_eval["prediccion"]),
                "MAPE": mape(df_eval["real"], df_eval["prediccion"]),
                "tiempo_segundos": tiempo
            })

metrics = pd.DataFrame(rows)

# Exportar tabla completa con valores exactos
metrics.to_csv(
    os.path.join(OUTPUT_DIR, "metricas_desacopladas_5057.csv"),
    index=False,
    encoding="utf-8-sig"
)

# =========================
# Crear versión redondeada para pegar en Word
# =========================
metrics_fmt = metrics.copy()
metrics_fmt["MAE (MW)"] = metrics_fmt["MAE"].apply(format_number)
metrics_fmt["RMSE (MW)"] = metrics_fmt["RMSE"].apply(format_number)
metrics_fmt["MAPE (%)"] = metrics_fmt["MAPE"].apply(format_number)
metrics_fmt["Tiempo de cómputo (s)"] = metrics_fmt["tiempo_segundos"].apply(format_number)

metrics_fmt = metrics_fmt[[
    "modelo",
    "ventana",
    "unidad",
    "MAE (MW)",
    "RMSE (MW)",
    "MAPE (%)",
    "Tiempo de cómputo (s)"
]]

metrics_fmt.to_csv(
    os.path.join(OUTPUT_DIR, "metricas_desacopladas_5057_redondeadas.csv"),
    index=False,
    encoding="utf-8-sig"
)

# Exportar una tabla por ventana
for window in [4, 12, 24]:
    temp = metrics_fmt[metrics_fmt["ventana"] == window].copy()
    temp.to_csv(
        os.path.join(OUTPUT_DIR, f"tabla_resultados_ventana_{window}_redondeada.csv"),
        index=False,
        encoding="utf-8-sig"
    )

print("\nMétricas desacopladas redondeadas:")
print(metrics_fmt.to_string(index=False))

print("\nArchivos generados:")
print("outputs_5057/metricas_desacopladas_5057.csv")
print("outputs_5057/metricas_desacopladas_5057_redondeadas.csv")
print("outputs_5057/tabla_resultados_ventana_4_redondeada.csv")
print("outputs_5057/tabla_resultados_ventana_12_redondeada.csv")
print("outputs_5057/tabla_resultados_ventana_24_redondeada.csv")