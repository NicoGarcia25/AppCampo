"""
=============================================================================
PrecioJusto Campo — Fase 2: Modelo de Precios
=============================================================================
Módulo A: Percentil histórico rodante (ventana 3 años)
Módulo B: Predicción Prophet a 30/60/90 días
Módulo C: Generación de señales de alerta (VENDER / NEUTRAL / ESPERAR)
Módulo D: Backtest de la estrategia (2022-2025)

Salida: predicciones.csv con todas las columnas listas para el dashboard
=============================================================================
"""

import logging
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
from prophet import Prophet

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────

RUTA_DATASET    = "dataset_maestro_granos.csv"
RUTA_SALIDA     = "predicciones.csv"

VENTANA_PERCENTIL = 365   # 18 meses
HORIZONTE_DIAS    = 120        # predecir hasta 90 días adelante

CULTIVOS = ["soja", "maiz", "trigo"]

# Umbrales de señal
UMBRAL_VENDER  = 65   # percentil > 70 → señal verde
UMBRAL_ESPERAR = 40   # percentil < 40 → señal roja


# =============================================================================
# MÓDULO A: PERCENTIL HISTÓRICO RODANTE
# =============================================================================

def calcular_percentil_rodante(serie: pd.Series, ventana: int = VENTANA_PERCENTIL) -> pd.Series:
    """
    Para cada día calcula en qué percentil está el precio actual
    respecto a los últimos `ventana` días.

    Ejemplo: percentil 78 significa que el precio de hoy es mayor
    al 78% de los precios de los últimos 3 años.

    Parámetros
    ----------
    serie   : pd.Series con precios diarios (index = fechas)
    ventana : int, cantidad de días hacia atrás a considerar

    Retorna
    -------
    pd.Series con percentil (0-100) para cada fecha
    """
    def percentil_en_fecha(x):
        # x es la ventana rodante — el último valor es el precio de hoy
        if len(x) < 30:  # mínimo 30 observaciones para que sea significativo
            return np.nan
        precio_hoy = x.iloc[-1]
        return round(float(np.sum(x < precio_hoy) / len(x) * 100), 1)

    log.info(f"  Calculando percentiles rodantes (ventana {ventana // 365} años)...")
    percentiles = serie.rolling(window=ventana, min_periods=30).apply(
        percentil_en_fecha, raw=False
    )
    return percentiles


# =============================================================================
# MÓDULO B: MODELO PROPHET
# =============================================================================

def entrenar_prophet(df: pd.DataFrame, col_precio: str) -> tuple:
    """
    Entrena un modelo Prophet sobre la serie de precios de un cultivo.

    Prophet maneja automáticamente:
      - Estacionalidad anual (ciclo agrícola)
      - Estacionalidad semanal (mercados cierran fin de semana)
      - Tendencia de largo plazo

    Parámetros
    ----------
    df         : DataFrame con columna 'fecha' y col_precio
    col_precio : nombre de la columna de precio (ej: 'precio_soja')

    Retorna
    -------
    (modelo Prophet entrenado, DataFrame de forecast)
    """
    cultivo = col_precio.replace("precio_", "")
    log.info(f"  Entrenando Prophet para {cultivo}...")

    # Prophet requiere columnas exactas 'ds' (fecha) y 'y' (valor)
    df_prophet = (
        df[["fecha", col_precio]]
        .dropna()
        .rename(columns={"fecha": "ds", col_precio: "y"})
        .sort_values("ds")
    )

    # Eliminar outliers extremos (> 3 desviaciones estándar)
    media = df_prophet["y"].mean()
    std   = df_prophet["y"].std()
    df_prophet = df_prophet[
        (df_prophet["y"] >= media - 3 * std) &
        (df_prophet["y"] <= media + 3 * std)
    ]

    # Configuración del modelo
    modelo = Prophet(
        changepoint_prior_scale=0.05,    # sensibilidad a cambios de tendencia
        seasonality_prior_scale=10,      # fuerza de estacionalidades
        yearly_seasonality=True,         # ciclo agrícola anual
        weekly_seasonality=True,         # precios bajan fin de semana
        daily_seasonality=False,         # no hay patrón diario intra-día
        interval_width=0.80,             # intervalo de confianza del 80%
    )

    # Estacionalidad de campaña argentina (semestral)
    modelo.add_seasonality(
        name="campania_arg",
        period=182.5,       # semestral
        fourier_order=5,
        mode="multiplicative",
    )

    # Suprimir output de Stan durante el entrenamiento
    import logging as pylog
    pylog.getLogger("prophet").setLevel(pylog.WARNING)
    pylog.getLogger("cmdstanpy").setLevel(pylog.WARNING)

    modelo.fit(df_prophet)

    # Generar fechas futuras (solo días hábiles de mercado)
    futuro = modelo.make_future_dataframe(
    periods=HORIZONTE_DIAS + 90,
    freq="B"
)
    forecast = modelo.predict(futuro)

    log.info(f"  ✓ Prophet {cultivo}: entrenado sobre {len(df_prophet)} observaciones")
    return modelo, forecast


def extraer_predicciones(forecast: pd.DataFrame, col_cultivo: str) -> pd.DataFrame:
    """
    Extrae del forecast de Prophet las columnas relevantes
    y las renombra con el prefijo del cultivo.
    """
    cols = {
        "ds":            "fecha",
        "yhat":          f"pred_{col_cultivo}",
        "yhat_lower":    f"pred_{col_cultivo}_lower",
        "yhat_upper":    f"pred_{col_cultivo}_upper",
        "trend":         f"trend_{col_cultivo}",
    }
    df_pred = forecast[list(cols.keys())].rename(columns=cols)
    df_pred["fecha"] = pd.to_datetime(df_pred["fecha"])
    return df_pred


# =============================================================================
# MÓDULO C: SEÑALES DE ALERTA
# =============================================================================

def generar_senal(percentil: float, pred_30d: float, precio_actual: float) -> str:
    """
    Combina el percentil histórico con la dirección de la predicción
    para generar una señal de trading.

    Lógica:
      VENDER  → percentil alto (precio caro vs historia) Y tendencia neutra/alcista
      ESPERAR → percentil bajo (precio barato, puede subir)
      NEUTRAL → zona intermedia o señales contradictorias

    Parámetros
    ----------
    percentil     : percentil actual (0-100)
    pred_30d      : precio predicho en 30 días
    precio_actual : precio de hoy

    Retorna
    -------
    str: "VENDER" | "NEUTRAL" | "ESPERAR"
    """
    if pd.isna(percentil) or pd.isna(pred_30d):
        return "SIN DATOS"

    cambio_esperado = (pred_30d - precio_actual) / precio_actual * 100

    if percentil >= UMBRAL_VENDER:
        # Precio históricamente alto
        if cambio_esperado <= 5:
            # No se espera suba importante → mejor vender ahora
            return "VENDER"
        else:
            # Se espera suba fuerte → mantener un poco más
            return "NEUTRAL"

    elif percentil <= UMBRAL_ESPERAR:
        # Precio históricamente bajo → esperar recuperación
        return "ESPERAR"

    else:
        # Zona intermedia
        return "NEUTRAL"


def color_senal(senal: str) -> str:
    """Retorna el color semáforo de la señal para el dashboard."""
    return {"VENDER": "verde", "NEUTRAL": "amarillo", "ESPERAR": "rojo"}.get(senal, "gris")


# =============================================================================
# MÓDULO D: BACKTEST
# =============================================================================

def backtest_estrategia(df: pd.DataFrame, col_precio: str, col_percentil: str) -> dict:
    """
    Simula cuánto hubiera ganado un productor que vendió cada vez
    que el sistema emitió señal VENDER vs vender en fecha aleatoria.

    Período de backtest: 2022-01-01 hasta hoy (datos reales).

    Retorna un dict con métricas del backtest para mostrar en el dashboard.
    """
    cultivo = col_precio.replace("precio_", "")
    log.info(f"  Corriendo backtest para {cultivo}...")

    df_bt = df[["fecha", col_precio, col_percentil]].dropna().copy()
    df_bt = df_bt[df_bt["fecha"] >= "2022-01-01"].sort_values("fecha")

    if df_bt.empty:
        return {}

    # Estrategia del modelo: vender cuando percentil > umbral
    ventas_modelo = df_bt[df_bt[col_percentil] >= UMBRAL_VENDER][col_precio]
    precio_modelo = ventas_modelo.mean() if not ventas_modelo.empty else 0

    # Estrategia base: vender en cualquier día hábil (promedio del período)
    precio_promedio = df_bt[col_precio].mean()

    # Mejora porcentual
    mejora = ((precio_modelo - precio_promedio) / precio_promedio * 100) if precio_promedio > 0 else 0

    # Cantidad de señales VENDER emitidas
    n_senales = len(ventas_modelo)

    resultado = {
        "cultivo":         cultivo,
        "precio_modelo":   round(precio_modelo, 2),
        "precio_promedio": round(precio_promedio, 2),
        "mejora_pct":      round(mejora, 2),
        "n_senales":       n_senales,
        "periodo":         f"{df_bt['fecha'].min().date()} → {df_bt['fecha'].max().date()}",
    }

    log.info(f"  ✓ Backtest {cultivo}: +{mejora:.1f}% vs vender al azar ({n_senales} señales)")
    return resultado


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def correr_pipeline() -> pd.DataFrame:
    """
    Orquesta los módulos A, B, C y D para todos los cultivos.
    Genera el archivo predicciones.csv listo para el dashboard.
    """
    log.info("=" * 60)
    log.info("PrecioJusto Campo — Fase 2: Modelo de Precios")
    log.info("=" * 60)

    # ── Cargar dataset maestro ────────────────────────────────────────────────
    if not Path(RUTA_DATASET).exists():
        raise FileNotFoundError(
            f"No se encontró {RUTA_DATASET}. "
            "Ejecutar primero fase1_ingesta_granos.py y patch_stocks_mundiales.py"
        )

    log.info(f"Cargando dataset: {RUTA_DATASET}")
    df = pd.read_csv(RUTA_DATASET, parse_dates=["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)
    log.info(f"  Filas: {len(df):,} | Rango: {df['fecha'].min().date()} → {df['fecha'].max().date()}")

    # DataFrame que acumula todos los resultados
    df_resultado = df.copy()

    # Resultados de backtest (para exportar aparte)
    backtests = []

    for cultivo in CULTIVOS:
        col_precio = f"precio_{cultivo}"

        if col_precio not in df.columns:
            log.warning(f"Columna {col_precio} no encontrada, saltando...")
            continue

        log.info(f"\n── Procesando {cultivo.upper()} ──────────────────────────")

          # ── Módulo A: Percentiles ─────────────────────────────────────────────
        serie_precios = df.set_index("fecha")[col_precio]
        percentiles = calcular_percentil_rodante(serie_precios)
        percentiles.name = f"percentil_{cultivo}"

        # Fix: reset_index() con DatetimeIndex genera "fecha" directamente
        df_perc = percentiles.reset_index()
        if "fecha" not in df_perc.columns:
            df_perc = df_perc.rename(columns={df_perc.columns[0]: "fecha"})
        df_perc["fecha"] = pd.to_datetime(df_perc["fecha"])

        # Eliminar columna si ya existe (evita duplicados en re-runs)
        if f"percentil_{cultivo}" in df_resultado.columns:
            df_resultado = df_resultado.drop(columns=[f"percentil_{cultivo}"])

        df_resultado = df_resultado.merge(df_perc, on="fecha", how="outer")

        # ── Módulo B: Prophet ─────────────────────────────────────────────────
        modelo, forecast = entrenar_prophet(df, col_precio)
        df_pred = extraer_predicciones(forecast, cultivo)
        df_resultado = df_resultado.merge(df_pred, on="fecha", how="outer")

        # ── Módulo C: Señales ─────────────────────────────────────────────────
        log.info(f"  Generando señales para {cultivo}...")

        # Construir lookup de predicción a 30 días: para cada fecha, qué predice el modelo 30 días después
        df_pred_30 = df_pred[["fecha", f"pred_{cultivo}"]].copy()
        df_pred_30["fecha_base"] = df_pred_30["fecha"] - pd.Timedelta(days=30)
        df_pred_30 = df_pred_30.rename(columns={f"pred_{cultivo}": f"pred_{cultivo}_30d"})
        df_pred_30 = df_pred_30[["fecha_base", f"pred_{cultivo}_30d"]].rename(
            columns={"fecha_base": "fecha"}
        )
        df_resultado = df_resultado.merge(df_pred_30, on="fecha", how="outer")

        # Aplicar señal fila por fila
        df_resultado[f"senal_{cultivo}"] = df_resultado.apply(
            lambda row: generar_senal(
                percentil=row.get(f"percentil_{cultivo}"),
                pred_30d=row.get(f"pred_{cultivo}_30d"),
                precio_actual=row.get(col_precio),
            ),
            axis=1,
        )

        # Precio en ARS (usando TC blue)
        if "tipo_cambio_blue" in df_resultado.columns:
            df_resultado[f"precio_{cultivo}_ars"] = (
                df_resultado[col_precio] * df_resultado["tipo_cambio_blue"]
            ).round(0)

        # ── Módulo D: Backtest ────────────────────────────────────────────────
        bt = backtest_estrategia(df_resultado, col_precio, f"percentil_{cultivo}")
        if bt:
            backtests.append(bt)

    # ── Exportar resultados ───────────────────────────────────────────────────
    df_resultado.to_csv(RUTA_SALIDA, index=False, encoding="utf-8")
    log.info(f"\n✓ Predicciones exportadas: {RUTA_SALIDA}")
    log.info(f"  Columnas generadas: {len(df_resultado.columns)}")

    # Exportar backtest
    if backtests:
        df_bt = pd.DataFrame(backtests)
        df_bt.to_csv("backtest_resultados.csv", index=False)
        log.info(f"✓ Backtest exportado: backtest_resultados.csv")

        log.info("\n" + "─" * 55)
        log.info("RESUMEN DE BACKTEST (2022–2025)")
        for bt in backtests:
            log.info(
                f"  {bt['cultivo'].upper():<6} | "
                f"Modelo: ${bt['precio_modelo']:.0f}/tn | "
                f"Base: ${bt['precio_promedio']:.0f}/tn | "
                f"Mejora: +{bt['mejora_pct']:.1f}% | "
                f"Señales: {bt['n_senales']}"
            )
        log.info("─" * 55)

    # ── Preview del estado actual ─────────────────────────────────────────────
    log.info("\nÚLTIMO REGISTRO DEL MODELO (hoy):")
    ultima = df_resultado.dropna(subset=["percentil_soja"]).iloc[-1]
    for cultivo in CULTIVOS:
        precio = ultima.get(f"precio_{cultivo}", "N/A")
        perc   = ultima.get(f"percentil_{cultivo}", "N/A")
        senal  = ultima.get(f"senal_{cultivo}", "N/A")
        pred30 = ultima.get(f"pred_{cultivo}_30d", "N/A")
        log.info(
            f"  {cultivo.upper():<6} | "
            f"Precio: ${precio:.0f}/tn | "
            f"Percentil: {perc:.0f}° | "
            f"Pred 30d: ${pred30:.0f}/tn | "
            f"Señal: {senal}"
        )

    log.info("\n✅ Fase 2 completada. Ejecutar dashboard.py para visualizar.")
    return df_resultado


if __name__ == "__main__":
    correr_pipeline()
