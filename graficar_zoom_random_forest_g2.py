import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Leer archivo de predicciones
df = pd.read_csv("outputs_5057/predicciones_modelos_5057.csv")

# Filtrar modelo Random Forest, ventana 12 y unidad G2
df = df[
    (df["modelo"] == "Random Forest") &
    (df["ventana"] == 12) &
    (df["unidad"] == "G2")
].copy()

# Convertir fecha
df["fecha_hora"] = pd.to_datetime(df["fecha_hora"])

# Filtrar tramo crítico
inicio = "2025-03-30 18:00:00"
fin = "2025-03-31 23:00:00"

df_zoom = df[
    (df["fecha_hora"] >= inicio) &
    (df["fecha_hora"] <= fin)
].copy()

# Función para cortar líneas cuando hay saltos temporales grandes
def cortar_saltos_temporales(data, columna_valor, max_minutos=30):
    data = data.sort_values("fecha_hora").copy()
    data["delta_min"] = data["fecha_hora"].diff().dt.total_seconds() / 60
    data.loc[data["delta_min"] > max_minutos, columna_valor] = np.nan
    return data

# Valor real y predicción con cortes temporales
real = df_zoom[["fecha_hora", "real"]].drop_duplicates().sort_values("fecha_hora")
real = cortar_saltos_temporales(real, "real")

pred = df_zoom[["fecha_hora", "prediccion"]].drop_duplicates().sort_values("fecha_hora")
pred = cortar_saltos_temporales(pred, "prediccion")

# Crear gráfico
plt.figure(figsize=(12, 6))

plt.plot(
    real["fecha_hora"],
    real["real"],
    marker="o",
    markersize=4,
    linewidth=2,
    label="Valor real G2"
)

plt.plot(
    pred["fecha_hora"],
    pred["prediccion"],
    marker="o",
    markersize=4,
    linewidth=1.5,
    label="Random Forest"
)

plt.title("Zoom del tramo crítico en G2 - Random Forest")
plt.xlabel("Fecha y hora")
plt.ylabel("Potencia efectiva (MW)")
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()

# Guardar imagen corregida
plt.savefig("outputs_5057/zoom_random_forest_g2_tramo_critico_corregido.png", dpi=300)
plt.close()

print("Imagen generada en: outputs_5057/zoom_random_forest_g2_tramo_critico_corregido.png")