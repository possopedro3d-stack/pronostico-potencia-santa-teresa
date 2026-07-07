import pandas as pd
import matplotlib.pyplot as plt

# Leer archivo de predicciones
df = pd.read_csv("outputs_5057/predicciones_modelos_5057.csv")

# Convertir fecha
df["fecha_hora"] = pd.to_datetime(df["fecha_hora"])

# Filtrar XGBoost con ventana 12
df = df[(df["modelo"] == "XGBoost") & (df["ventana"] == 12)].copy()

# Separar por unidad
g1 = df[df["unidad"] == "G1"].sort_values("fecha_hora")
g2 = df[df["unidad"] == "G2"].sort_values("fecha_hora")

# Crear figura
plt.figure(figsize=(16, 8))

# G1
plt.plot(g1["fecha_hora"], g1["real"], label="Valor real G1", linewidth=2)
plt.plot(g1["fecha_hora"], g1["prediccion"], linestyle="--", label="XGBoost G1", linewidth=2)

# G2
plt.plot(g2["fecha_hora"], g2["real"], label="Valor real G2", linewidth=2)
plt.plot(g2["fecha_hora"], g2["prediccion"], linestyle="--", label="XGBoost G2", linewidth=2)

# Título y ejes
plt.title("XGBoost - comparación con zoom en eje Y", fontsize=20)
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
ruta_salida = "outputs_5057/xgboost_zoom_eje_y_w12.png"
plt.savefig(ruta_salida, dpi=300, bbox_inches="tight")
plt.close()

print(f"Imagen generada en: {ruta_salida}")