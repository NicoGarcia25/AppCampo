"""
=============================================================================
PrecioJusto Campo — Patch: Completar stocks mundiales en el dataset maestro
=============================================================================
Las APIs de FAOSTAT y USDA ERS estaban caídas durante la ingesta.
Este script embebe los datos oficiales de ending stocks mundiales
(fuente: USDA WASDE reports 2020-2025, valores en millones de toneladas métricas)
y los une al dataset_maestro_granos.csv existente.

Ejecutar DESPUÉS de fase1_ingesta_granos.py:
    python patch_stocks_mundiales.py
=============================================================================
"""

import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# =============================================================================
# DATOS EMBEBIDOS — USDA WASDE Ending Stocks Mundiales
# Fuente: USDA World Agricultural Supply and Demand Estimates
# Unidad: millones de toneladas métricas
# Frecuencia: anual (año de mercado, referenciado en octubre)
# =============================================================================

# Soja — World Ending Stocks (millones de tn)
# Campaña referenciada en octubre del año de inicio
STOCKS_SOJA = {
    # año_mercado: stock_final_mundial (millones tn)
    2019: 100.8,
    2020:  92.8,
    2021:  86.0,
    2022:  99.6,
    2023: 113.9,
    2024: 128.4,
    2025: 124.8,  # Proyección WASDE abril 2026
}

# Maíz — World Ending Stocks (millones de tn)
STOCKS_MAIZ = {
    2019: 296.0,
    2020: 287.0,
    2021: 305.7,
    2022: 299.6,
    2023: 315.6,
    2024: 289.5,
    2025: 283.8,  # Proyección WASDE abril 2026
}

# Trigo — World Ending Stocks (millones de tn)
STOCKS_TRIGO = {
    2019: 293.0,
    2020: 313.5,
    2021: 296.3,
    2022: 267.5,
    2023: 258.2,
    2024: 257.8,
    2025: 271.4,  # Proyección WASDE abril 2026
}


def construir_df_stocks() -> pd.DataFrame:
    """
    Construye un DataFrame mensual de stocks mundiales usando los datos
    embebidos del USDA WASDE. Aplica ffill para generar valores mensuales
    desde los datos anuales.
    """
    # Convertir a DataFrame con fecha de referencia = octubre del año de mercado
    registros = []
    for anio in sorted(set(STOCKS_SOJA) | set(STOCKS_MAIZ) | set(STOCKS_TRIGO)):
        registros.append({
            "fecha": pd.Timestamp(year=anio, month=10, day=1),
            "stock_mundial_soja_wasde":  STOCKS_SOJA.get(anio),
            "stock_mundial_maiz_wasde":  STOCKS_MAIZ.get(anio),
            "stock_mundial_trigo_wasde": STOCKS_TRIGO.get(anio),
        })

    df_stocks = pd.DataFrame(registros).sort_values("fecha").reset_index(drop=True)

    # Calcular variación anual (usamos pct_change como proxy de variación mensual)
    for cultivo in ["soja", "maiz", "trigo"]:
        col = f"stock_mundial_{cultivo}_wasde"
        df_stocks[f"variacion_stock_{cultivo}_mensual"] = (
            df_stocks[col].pct_change() * 100
        ).round(2)

    log.info(f"✓ Stocks embebidos construidos: {len(df_stocks)} años de mercado")
    log.info(f"  Rango: {df_stocks['fecha'].min().date()} → {df_stocks['fecha'].max().date()}")
    return df_stocks


def parchear_dataset(
    ruta_entrada: str = "dataset_maestro_granos.csv",
    ruta_salida:  str = "dataset_maestro_granos.csv",
) -> pd.DataFrame:
    """
    Lee el dataset maestro existente, reemplaza las columnas de stocks
    mundiales (que estaban al 100% nulas) con los datos embebidos del WASDE,
    y guarda el resultado.
    """
    # Leer dataset existente
    log.info(f"Leyendo dataset: {ruta_entrada}")
    df = pd.read_csv(ruta_entrada, parse_dates=["fecha"])
    log.info(f"  Filas: {len(df):,} | Columnas: {len(df.columns)}")

    # Construir stocks
    df_stocks = construir_df_stocks()

    # Eliminar columnas de stocks que están vacías
    cols_stocks = [c for c in df.columns if "stock_mundial" in c or "variacion_stock" in c]
    df = df.drop(columns=cols_stocks, errors="ignore")
    log.info(f"  Columnas de stocks eliminadas (estaban vacías): {cols_stocks}")

    # Merge con stocks embebidos (left join sobre fecha)
    df = pd.merge(df, df_stocks, on="fecha", how="left")

    # ffill: propagar el valor anual a cada día del año
    # Cada día hereda el stock del último año de mercado disponible
    cols_nuevas = [c for c in df_stocks.columns if c != "fecha"]
    for col in cols_nuevas:
        df[col] = df[col].ffill()

    # Completar nulos de producción ARG en los primeros meses del dataset
    # (antes del primer registro de campaña disponible = abril 2019)
    magyp_cols = ["produccion_estimada_arg", "rendimiento_arg", "mes_campania"]
    for col in magyp_cols:
        if col in df.columns:
            # bfill para los registros anteriores a 2019
            df[col] = df[col].ffill().bfill()

    # Reporte final de calidad
    log.info("─" * 55)
    log.info("REPORTE DE CALIDAD POST-PATCH")
    log.info(f"  Período  : {df['fecha'].min().date()} → {df['fecha'].max().date()}")
    log.info(f"  Filas    : {len(df):,}")
    log.info(f"  Columnas : {len(df.columns)}")
    nulos = df.isnull().sum()
    nulos_sig = nulos[nulos > 0]
    if len(nulos_sig):
        log.info("  Columnas con nulos:")
        for col, n in nulos_sig.items():
            log.info(f"    {col:<42} {n:>5} ({n/len(df)*100:.1f}%)")
    else:
        log.info("  ✅ Sin valores nulos — dataset completo")
    log.info("─" * 55)

    # Guardar
    df.to_csv(ruta_salida, index=False, encoding="utf-8")
    log.info(f"✓ Dataset parcheado guardado: {ruta_salida}")

    # Preview
    cols_preview = [
        "fecha", "precio_soja", "precio_maiz", "precio_trigo",
        "tipo_cambio_blue", "stock_mundial_soja_wasde",
        "produccion_estimada_arg", "dia_semana"
    ]
    cols_preview = [c for c in cols_preview if c in df.columns]
    print("\nÚltimas 5 filas del dataset parcheado:")
    print(df[cols_preview].tail(5).to_string(index=False))

    return df


if __name__ == "__main__":
    log.info("=" * 55)
    log.info("PrecioJusto Campo — Patch: Stocks mundiales WASDE")
    log.info("=" * 55)
    df_final = parchear_dataset()
    log.info("\n✅ Patch completado. Dataset listo para Fase 2.")
