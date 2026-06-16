import pandas as pd

# Cargo los datos de los 3 modelos
df_eff = pd.read_csv("./efficientnet/Explicabilidad/bloque0_efficientnet.csv")
df_res = pd.read_csv("./resnet/Explicabilidad/bloque0_resnet.csv")
df_vit = pd.read_csv("./explicabilidad_vit/bloque0_vit.csv")

# Selecciono las columnas relevantes de cada modelo con sufijos para distinguirlas
df_merge = df_eff[["imagen_id", "imagen_path", "label_real", "label_real_nombre",
                    "acierto", "conf_correcta", "pred_efficient", "pred_nombre"]] \
    .rename(columns={
        "acierto"       : "acierto_eff",
        "conf_correcta" : "conf_correcta_eff",
        "pred_efficient": "pred_eff",
        "pred_nombre"   : "pred_nombre_eff"
    }) \
    .merge(
        df_res[["imagen_id", "acierto", "conf_correcta", "pred_resnet", "pred_nombre"]].rename(columns={
            "acierto"       : "acierto_res",
            "conf_correcta" : "conf_correcta_res",
            "pred_resnet"   : "pred_res",
            "pred_nombre"   : "pred_nombre_res"
        }),
        on="imagen_id"
    ) \
    .merge(
        df_vit[["imagen_id", "acierto", "conf_correcta", "pred_vit", "pred_nombre"]].rename(columns={
            "acierto"       : "acierto_vit",
            "conf_correcta" : "conf_correcta_vit",
            "pred_vit"      : "pred_vit",
            "pred_nombre"   : "pred_nombre_vit"
        }),
        on="imagen_id"
    )

# Aciertos comunes — los 3 aciertan
df_todos_aciertan = df_merge[
    (df_merge["acierto_eff"] == 1) &
    (df_merge["acierto_res"] == 1) &
    (df_merge["acierto_vit"] == 1)
]

candidatas_finales = (
    df_todos_aciertan
    .sample(frac=1, random_state=16)
    .groupby("label_real")
    .head(4)
    .reset_index(drop=True)
)

candidatas_finales.to_csv("candidatas_finales_comunes.csv", index=False)
print("\n=== ACIERTOS COMUNES ===")
print(candidatas_finales[[
    "imagen_id", "label_real_nombre",
    "conf_correcta_eff", "pred_nombre_eff",
    "conf_correcta_res", "pred_nombre_res",
    "conf_correcta_vit", "pred_nombre_vit"
]])

# Fallos comunes — los 3 fallan
df_todos_fallan = df_merge[
    (df_merge["acierto_eff"] == 0) &
    (df_merge["acierto_res"] == 0) &
    (df_merge["acierto_vit"] == 0)
]

fallos_comunes = (
    df_todos_fallan
    .sample(frac=1, random_state=16)
    .groupby("label_real")
    .head(4)
    .reset_index(drop=True)
)

fallos_comunes.to_csv("fallos_comunes.csv", index=False)
print("\n=== FALLOS COMUNES ===")
print(fallos_comunes[[
    "imagen_id", "label_real_nombre",
    "conf_correcta_eff", "pred_nombre_eff",
    "conf_correcta_res", "pred_nombre_res",
    "conf_correcta_vit", "pred_nombre_vit"
]])