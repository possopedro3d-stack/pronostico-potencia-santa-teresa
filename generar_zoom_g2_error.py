import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = "outputs_5057"

preds_old = pd.read_csv(os.path.join(OUTPUT_DIR, "predicciones_modelos_5057.csv"))
preds_lstm = pd.read_csv(os.path.join(OUTPUT_DIR, "predicciones_lstm_g1_g2_5057.csv"))

preds_old = preds_old[preds_old["modelo"] != "LSTM"].copy()
preds = pd.concat([preds_old, preds_lstm], ignore_index=True)

preds["fecha_hora"] = pd.to_datetime(preds["fecha_hora"])

df = preds[
    (preds["unidad"] == "G2") &
    (preds["fecha_hora"] >= "2025-03-30 18:00:00") &
    (preds["fecha_hora"] <= "2025-03-31 23:00:00")
].copy()

def cortar_saltos_temporales(data, columna_valor, max_minutos=30):
    data = data.sort_values("fecha_hora").copy()
    data["delta_min"] = data["fecha_hora"].diff().dt.total_seconds() / 60
    data.loc[data["delta_min"] > max_minutos, columna_valor] = np.nan
    return data

real = df[["fecha_hora", "real"]].drop_duplicates().sort_values("fecha_hora")
real = cortar_saltos_temporales(real, "real")

configs = {
    "Baseline persistencia": 4,
    "Random Forest": 12,
    "XGBoost": 12,
    "LSTM": 12,
}

plt.figure(figsize=(12, 6))

plt.plot(
    real["fecha_hora"],
    real["real"],
    marker="o",
    markersize=3,
    linewidth=2,
    label="Valor real G2"
)

for modelo, ventana in configs.items():
    temp = df[(df["modelo"] == modelo) & (df["ventana"] == ventana)].copy()
    temp = cortar_saltos_temporales(temp.sort_values("fecha_hora"), "prediccion")
    plt.plot(
        temp["fecha_hora"],
        temp["prediccion"],
        marker="o",
        markersize=2,
        linewidth=1.5,
        label=f"{modelo}"
    )

plt.title("Zoom del tramo crítico en G2")
plt.xlabel("Fecha y hora")
plt.ylabel("Potencia efectiva (MW)")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

output_path = os.path.join(OUTPUT_DIR, "zoom_tramo_critico_g2_30_31_marzo_corregido.png")
plt.savefig(output_path, dpi=300)
plt.close()

print("Gráfico generado correctamente:")
print(output_path)