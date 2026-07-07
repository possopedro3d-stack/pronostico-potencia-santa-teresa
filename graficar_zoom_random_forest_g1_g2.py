import pandas as pd
import matplotlib.pyplot as plt
import os

# Cargar predicciones
df = pd.read_csv("outputs_5057/predicciones_modelos_5057.csv")

# Filtrar Random Forest con ventana 12
df = df[
    (df["modelo"] == "Random Forest") &
    (df["ventana"] == 12)
].copy()

# Convertir fecha
df["fecha_hora"] = pd.to_datetime(df["fecha_hora"])

# Separar por unidad
g1 = df[df["unidad"] == "G1"].sort_values("fecha_hora")
g2 = df[df["unidad"] == "G2"].sort_values("fecha_hora")

# Crear figura
plt.figure(figsize=(14, 6))

# G1
plt.plot(g1["fecha_hora"], g1["real"], label="Valor real G1", linewidth=2)
plt.plot(g1["fecha_hora"], g1["prediccion"], linestyle="--", label="Random Forest G1", linewidth=2)

# G2
plt.plot(g2["fecha_hora"], g2["real"], label="Valor real G2", linewidth=2)
plt.plot(g2["fecha_hora"], g2["prediccion"], linestyle="--", label="Random Forest G2", linewidth=2)

# Título y etiquetas
plt.title("Random Forest - comparación con zoom en eje Y")
plt.xlabel("Fecha y hora")
plt.ylabel("Potencia efectiva (MW)")

# Zoom en eje Y
plt.ylim(9, 13)

plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

# Guardar imagen
os.makedirs("outputs_5057", exist_ok=True)
ruta_salida = "outputs_5057/zoom_y_random_forest_g1_g2_w12.png"
plt.savefig(ruta_salida, dpi=300, bbox_inches="tight")
plt.close()

print(f"Imagen generada en: {ruta_salida}")