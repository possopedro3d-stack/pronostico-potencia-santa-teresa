import pandas as pd
import matplotlib.pyplot as plt

# Leer archivo de predicciones
df = pd.read_csv("outputs_5057/predicciones_modelos_5057.csv")

# Filtrar modelo baseline, ventana 4 y unidad G2
df = df[
    (df["modelo"] == "Baseline persistencia") &
    (df["ventana"] == 4) &
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

# Crear gráfico
plt.figure(figsize=(12, 6))

plt.plot(
    df_zoom["fecha_hora"],
    df_zoom["real"],
    marker="o",
    linewidth=2,
    label="Valor real G2"
)

plt.plot(
    df_zoom["fecha_hora"],
    df_zoom["prediccion"],
    marker="o",
    linewidth=1.5,
    label="Baseline persistencia"
)

plt.title("Zoom del tramo crítico en G2 - Baseline persistencia")
plt.xlabel("Fecha y hora")
plt.ylabel("Potencia efectiva (MW)")
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()

# Guardar imagen
plt.savefig("outputs_5057/zoom_baseline_g2_tramo_critico.png", dpi=300)
plt.close()

print("Imagen generada en: outputs_5057/zoom_baseline_g2_tramo_critico.png")