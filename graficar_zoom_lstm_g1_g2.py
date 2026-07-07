import pandas as pd
import matplotlib.pyplot as plt
import os

# Cargar predicciones LSTM corregidas con G1 y G2
df = pd.read_csv("outputs_5057/predicciones_lstm_g1_g2_5057.csv")

# Convertir fecha
df["fecha_hora"] = pd.to_datetime(df["fecha_hora"])

# Filtrar LSTM con ventana 12
df = df[
    (df["modelo"] == "LSTM") &
    (df["ventana"] == 12)
].copy()

# Separar por unidad
g1 = df[df["unidad"] == "G1"].sort_values("fecha_hora")
g2 = df[df["unidad"] == "G2"].sort_values("fecha_hora")

# Crear figura
plt.figure(figsize=(16, 8))

# G1
plt.plot(g1["fecha_hora"], g1["real"], label="Valor real G1", linewidth=2)
plt.plot(g1["fecha_hora"], g1["prediccion"], linestyle="--", label="LSTM G1", linewidth=2)

# G2
plt.plot(g2["fecha_hora"], g2["real"], label="Valor real G2", linewidth=2)
plt.plot(g2["fecha_hora"], g2["prediccion"], linestyle="--", label="LSTM G2", linewidth=2)

# Título y ejes
plt.title("LSTM - comparación con zoom en eje Y", fontsize=20)
plt.xlabel("Fecha y hora", fontsize=16)
plt.ylabel("Potencia efectiva (MW)", fontsize=16)

# Zoom en eje Y
plt.ylim(9, 13)

# Leyenda y rejilla
plt.legend(fontsize=14)
plt.grid(True, alpha=0.5)
plt.xticks(rotation=45)
plt.tight_layout()

# Guardar imagen
ruta_salida = "outputs_5057/lstm_zoom_eje_y_w12.png"
plt.savefig(ruta_salida, dpi=300, bbox_inches="tight")
plt.close()

print(f"Imagen generada en: {ruta_salida}")