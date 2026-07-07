import os
import pandas as pd

OUTPUT_DIR = "outputs_5057"

preds_old = pd.read_csv(os.path.join(OUTPUT_DIR, "predicciones_modelos_5057.csv"))
preds_lstm = pd.read_csv(os.path.join(OUTPUT_DIR, "predicciones_lstm_g1_g2_5057.csv"))

# Quitar LSTM antiguo porque solo tenía G2
preds_no_lstm = preds_old[preds_old["modelo"] != "LSTM"].copy()

# Unir predicciones corregidas
preds = pd.concat([preds_no_lstm, preds_lstm], ignore_index=True)

preds["fecha_hora"] = pd.to_datetime(preds["fecha_hora"])
preds["error_absoluto"] = (preds["real"] - preds["prediccion"]).abs()
preds["error_porcentual"] = preds["error_absoluto"] / preds["real"].abs() * 100

# Top 20 errores generales
top20 = preds.sort_values("error_absoluto", ascending=False).head(20)

# Top 10 errores por modelo
top_por_modelo = (
    preds.sort_values("error_absoluto", ascending=False)
    .groupby("modelo")
    .head(10)
    .sort_values(["modelo", "error_absoluto"], ascending=[True, False])
)

# Top 10 errores por unidad
top_por_unidad = (
    preds.sort_values("error_absoluto", ascending=False)
    .groupby("unidad")
    .head(10)
    .sort_values(["unidad", "error_absoluto"], ascending=[True, False])
)

top20.to_csv(os.path.join(OUTPUT_DIR, "analisis_errores_top20_desacoplado_5057.csv"), index=False, encoding="utf-8-sig")
top_por_modelo.to_csv(os.path.join(OUTPUT_DIR, "analisis_errores_top10_por_modelo_5057.csv"), index=False, encoding="utf-8-sig")
top_por_unidad.to_csv(os.path.join(OUTPUT_DIR, "analisis_errores_top10_por_unidad_5057.csv"), index=False, encoding="utf-8-sig")

print("\nTop 20 errores generales:")
print(top20[["fecha_hora", "unidad", "modelo", "ventana", "real", "prediccion", "error_absoluto", "error_porcentual"]].to_string(index=False))

print("\nArchivos generados:")
print("outputs_5057/analisis_errores_top20_desacoplado_5057.csv")
print("outputs_5057/analisis_errores_top10_por_modelo_5057.csv")
print("outputs_5057/analisis_errores_top10_por_unidad_5057.csv")