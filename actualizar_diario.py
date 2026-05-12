"""
=============================================================================
PrecioJusto Campo — Actualizador Diario Incremental
=============================================================================
Lógica:
  1. Lee dataset_maestro_granos.csv existente
  2. Detecta la última fecha registrada
  3. Descarga SOLO los días faltantes (precio, TC)
  4. Agrega las filas nuevas al CSV sin tocar el historial
  5. Recalcula percentiles y señales sobre el dataset completo
  6. Sobreescribe predicciones.csv listo para el dashboard

NO re-entrena Prophet todos los días (demasiado lento).
Re-entrena solo los lunes o cuando hay más de 30 días nuevos acumulados.

Ejecutar manualmente o via scheduler:
    python actualizar_diario.py
=============================================================================
"""

import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import requests
import io
import time

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

RUTA_DATASET     = "dataset_maestro_granos.csv"
RUTA_PREDICCIONES = "predicciones.csv"
RUTA_MODELO_CACHE = "modelo_cache.pkl"   # Prophet guardado en disco
RUTA_LOG_UPDATES  = "actualizaciones.log"

UMBRAL_VENDER  = 70
UMBRAL_ESPERAR = 40
VENTANA_PERCENTIL = 365

BUSHEL_A_TN = {"soja": 36.744, "maiz": 39.368, "trigo": 36.744}
YAHOO_TICKERS = {"soja": "ZS=F", "maiz": "ZC=F", "trigo": "ZW=F"}


# =============================================================================
# DETECCIÓN DE DÍAS FALTANTES
# =============================================================================

def detectar_dias_faltantes(df: pd.DataFrame) -> list:
    """
    Compara la última fecha del dataset con hoy y retorna
    la lista de fechas hábiles que faltan agregar.
    """
    ultima_fecha = pd.to_datetime(df["fecha"].max()).date()
    hoy = datetime.today().date()

    if ultima_fecha >= hoy:
        log.info(f"Dataset ya está actualizado hasta {ultima_fecha}. Nada que hacer.")
        return []

    # Generar fechas hábiles faltantes (lunes a viernes)
    fechas_faltantes = pd.bdate_range(
        start=ultima_fecha + timedelta(days=1),
        end=hoy,
    ).tolist()

    log.info(f"Última fecha en dataset: {ultima_fecha}")
    log.info(f"Fechas hábiles faltantes: {len(fechas_faltantes)}")
    return fechas_faltantes


# =============================================================================
# DESCARGA DE DATOS NUEVOS
# =============================================================================

def descargar_precios_nuevos(fecha_desde: datetime, fecha_hasta: datetime) -> pd.DataFrame:
    """
    Descarga precios CBOT para el rango de fechas faltantes.
    Usa yfinance con fallback a Stooq.
    """
    desde_str = fecha_desde.strftime("%Y-%m-%d")
    hasta_str = (fecha_hasta + timedelta(days=1)).strftime("%Y-%m-%d")

    try:
        import yfinance as yf
        log.info(f"Descargando precios nuevos ({desde_str} → {hasta_str})...")
        dfs = []
        for cultivo, ticker in YAHOO_TICKERS.items():
            try:
                data = yf.download(
                    ticker,
                    start=desde_str,
                    end=hasta_str,
                    progress=False,
                    auto_adjust=True,
                )
                if data.empty:
                    continue
                df = data[["Close"]].reset_index()
                df.columns = ["fecha", f"precio_{cultivo}"]
                df["fecha"] = pd.to_datetime(df["fecha"]).dt.tz_localize(None)
                df[f"precio_{cultivo}"] = (
                    pd.to_numeric(df[f"precio_{cultivo}"], errors="coerce")
                    * BUSHEL_A_TN[cultivo]
                ).round(2)
                dfs.append(df)
                time.sleep(0.2)
            except Exception as e:
                log.warning(f"  Error Yahoo {cultivo}: {e}")

        if not dfs:
            raise ValueError("Sin datos de Yahoo Finance")

        merged = dfs[0]
        for df in dfs[1:]:
            merged = pd.merge(merged, df, on="fecha", how="outer")
        return merged.sort_values("fecha").reset_index(drop=True)

    except Exception as e:
        log.warning(f"Yahoo Finance falló: {e}. Usando Stooq...")
        return _descargar_precios_stooq(desde_str, fecha_hasta.strftime("%Y%m%d"))


def _descargar_precios_stooq(desde_str: str, hasta_str: str) -> pd.DataFrame:
    """Fallback: precios desde Stooq."""
    stooq_tickers = {"soja": "zs.f", "maiz": "zc.f", "trigo": "zw.f"}
    dfs = []
    for cultivo, ticker in stooq_tickers.items():
        url = f"https://stooq.com/q/d/l/?s={ticker}&d1={desde_str.replace('-','')}&d2={hasta_str}&i=d"
        try:
            resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text))
            df.columns = [c.lower().strip() for c in df.columns]
            fecha_col  = next((c for c in df.columns if "date" in c), None)
            precio_col = next((c for c in df.columns if "close" in c), None)
            if not fecha_col or not precio_col:
                continue
            df = df[[fecha_col, precio_col]].rename(
                columns={fecha_col: "fecha", precio_col: f"precio_{cultivo}"}
            )
            df["fecha"] = pd.to_datetime(df["fecha"])
            df[f"precio_{cultivo}"] = (
                pd.to_numeric(df[f"precio_{cultivo}"], errors="coerce")
                * BUSHEL_A_TN[cultivo]
            ).round(2)
            dfs.append(df.dropna())
            time.sleep(0.3)
        except Exception as e:
            log.error(f"  Stooq {cultivo}: {e}")

    if not dfs:
        return pd.DataFrame()
    merged = dfs[0]
    for df in dfs[1:]:
        merged = pd.merge(merged, df, on="fecha", how="outer")
    return merged.sort_values("fecha").reset_index(drop=True)


def descargar_tc_nuevo(fecha_desde: datetime, fecha_hasta: datetime) -> pd.DataFrame:
    """
    Descarga tipo de cambio para los días faltantes.
    argentinadatos.com devuelve siempre el histórico completo,
    así que filtramos al rango que nos interesa.
    """
    endpoints = {
        "oficial": "https://api.argentinadatos.com/v1/cotizaciones/dolares/oficial",
        "blue":    "https://api.argentinadatos.com/v1/cotizaciones/dolares/blue",
    }
    dfs_tc = []
    for nombre, url in endpoints.items():
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            df = pd.DataFrame(resp.json())
            df = df.rename(columns={"venta": f"tipo_cambio_{nombre}"})
            df["fecha"] = pd.to_datetime(df["fecha"])
            df[f"tipo_cambio_{nombre}"] = pd.to_numeric(
                df[f"tipo_cambio_{nombre}"], errors="coerce"
            )
            # Solo los días nuevos
            df = df[
                (df["fecha"] >= pd.Timestamp(fecha_desde)) &
                (df["fecha"] <= pd.Timestamp(fecha_hasta))
            ][["fecha", f"tipo_cambio_{nombre}"]].dropna()
            dfs_tc.append(df)
            time.sleep(0.2)
        except Exception as e:
            log.warning(f"  TC {nombre}: {e}")

    if len(dfs_tc) == 2:
        return pd.merge(dfs_tc[0], dfs_tc[1], on="fecha", how="outer").sort_values("fecha")
    elif dfs_tc:
        return dfs_tc[0]
    return pd.DataFrame()


# =============================================================================
# INTEGRACIÓN DE FILAS NUEVAS AL DATASET
# =============================================================================

def integrar_filas_nuevas(df_existente: pd.DataFrame, df_precios_nuevo: pd.DataFrame,
                           df_tc_nuevo: pd.DataFrame) -> pd.DataFrame:
    """
    Construye las filas nuevas con el mismo esquema que el dataset
    existente y las concatena al final.

    Para columnas que no se actualizan diariamente (produccion_arg,
    stocks_wasde, mes_campania) propaga el último valor conocido (ffill).
    """
    if df_precios_nuevo.empty:
        log.warning("Sin precios nuevos para integrar.")
        return df_existente

    # Merge precios + TC nuevo
    df_nuevo = pd.merge(df_precios_nuevo, df_tc_nuevo, on="fecha", how="left") \
               if not df_tc_nuevo.empty else df_precios_nuevo.copy()

    # Columnas que se propagan desde el último registro del dataset histórico
    ultimo = df_existente.iloc[-1]
    cols_propagar = [
        "produccion_estimada_arg", "rendimiento_arg", "mes_campania",
        "stock_mundial_soja_wasde", "stock_mundial_maiz_wasde", "stock_mundial_trigo_wasde",
        "variacion_stock_soja_mensual", "variacion_stock_maiz_mensual",
        "variacion_stock_trigo_mensual",
    ]
    for col in cols_propagar:
        if col in df_existente.columns and col not in df_nuevo.columns:
            df_nuevo[col] = ultimo.get(col, None)

    # TC: si faltan días (feriados) usar el último conocido
    for tc_col in ["tipo_cambio_oficial", "tipo_cambio_blue"]:
        if tc_col not in df_nuevo.columns:
            df_nuevo[tc_col] = None
        df_nuevo[tc_col] = df_nuevo[tc_col].fillna(ultimo.get(tc_col))

    # Features de calendario
    df_nuevo["fecha"] = pd.to_datetime(df_nuevo["fecha"])
    df_nuevo["dia_semana"] = df_nuevo["fecha"].dt.day_name()
    df_nuevo["is_weekend"] = df_nuevo["fecha"].dt.dayofweek >= 5

    # Precio en ARS
    for cultivo in ["soja", "maiz", "trigo"]:
        col_p = f"precio_{cultivo}"
        col_ars = f"precio_{cultivo}_ars"
        if col_p in df_nuevo.columns and "tipo_cambio_blue" in df_nuevo.columns:
            df_nuevo[col_ars] = (df_nuevo[col_p] * df_nuevo["tipo_cambio_blue"]).round(0)

    # Asegurar mismo orden de columnas que el dataset existente
    for col in df_existente.columns:
        if col not in df_nuevo.columns:
            df_nuevo[col] = None
    df_nuevo = df_nuevo[df_existente.columns]

    # Concatenar y eliminar duplicados por fecha
    df_combinado = pd.concat([df_existente, df_nuevo], ignore_index=True)
    df_combinado = df_combinado.drop_duplicates(subset=["fecha"], keep="last")
    df_combinado = df_combinado.sort_values("fecha").reset_index(drop=True)

    filas_agregadas = len(df_combinado) - len(df_existente)
    log.info(f"✓ {filas_agregadas} filas nuevas integradas al dataset.")
    return df_combinado


# =============================================================================
# RECÁLCULO DE PERCENTILES Y SEÑALES (sin re-entrenar Prophet)
# =============================================================================

def recalcular_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recalcula los percentiles rodantes para todas las fechas.
    Es rápido (~segundos) porque no involucra ML.
    """
    log.info("Recalculando percentiles rodantes...")
    for cultivo in ["soja", "maiz", "trigo"]:
        col_precio = f"precio_{cultivo}"
        col_perc   = f"percentil_{cultivo}"
        if col_precio not in df.columns:
            continue

        serie = df.set_index("fecha")[col_precio]

        def pct_en_fecha(x):
            if len(x) < 30:
                return np.nan
            return round(float(np.sum(x < x.iloc[-1]) / len(x) * 100), 1)

        percentiles = serie.rolling(
            window=VENTANA_PERCENTIL, min_periods=30
        ).apply(pct_en_fecha, raw=False)

        df[col_perc] = percentiles.values

    log.info("  ✓ Percentiles recalculados.")
    return df


def recalcular_senales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recalcula las señales VENDER/NEUTRAL/ESPERAR usando percentiles
    y la predicción Prophet guardada en predicciones.csv.

    Si predicciones.csv existe, usa las predicciones ya calculadas.
    Si no, genera señal solo con el percentil.
    """
    log.info("Recalculando señales...")

    # Cargar predicciones Prophet existentes si están disponibles
    pred_cache = {}
    if Path(RUTA_PREDICCIONES).exists():
        df_pred = pd.read_csv(RUTA_PREDICCIONES, parse_dates=["fecha"])
        for cultivo in ["soja", "maiz", "trigo"]:
            col = f"pred_{cultivo}_30d"
            if col in df_pred.columns:
                pred_cache[cultivo] = df_pred.set_index("fecha")[col]

    for cultivo in ["soja", "maiz", "trigo"]:
        col_precio = f"precio_{cultivo}"
        col_perc   = f"percentil_{cultivo}"
        col_senal  = f"senal_{cultivo}"

        if col_precio not in df.columns or col_perc not in df.columns:
            continue

        def _senal(row):
            perc   = row.get(col_perc)
            precio = row.get(col_precio)
            # Buscar predicción a 30d del cache si existe
            pred_30d = None
            if cultivo in pred_cache:
                try:
                    pred_30d = pred_cache[cultivo].get(row["fecha"])
                except Exception:
                    pass
            # Si no hay predicción, usar solo el percentil
            if pred_30d is None or pd.isna(pred_30d):
                if pd.isna(perc):
                    return "SIN DATOS"
                if perc >= UMBRAL_VENDER:
                    return "VENDER"
                elif perc <= UMBRAL_ESPERAR:
                    return "ESPERAR"
                return "NEUTRAL"
            # Con predicción: lógica completa
            cambio_esp = (pred_30d - precio) / precio * 100 if precio else 0
            if perc >= UMBRAL_VENDER:
                return "VENDER" if cambio_esp <= 5 else "NEUTRAL"
            elif perc <= UMBRAL_ESPERAR:
                return "ESPERAR"
            return "NEUTRAL"

        df[col_senal] = df.apply(_senal, axis=1)

    log.info("  ✓ Señales recalculadas.")
    return df


# =============================================================================
# DECISIÓN: ¿RE-ENTRENAR PROPHET?
# =============================================================================

def necesita_reentrenamiento(df: pd.DataFrame) -> bool:
    """
    Re-entrenar Prophet es costoso (~5 min). Solo lo hacemos cuando:
      - Es lunes (inicio de semana agrícola)
      - Hay más de 30 filas nuevas desde el último entrenamiento
      - El archivo predicciones.csv no existe

    Retorna True si hay que re-entrenar.
    """
    if not Path(RUTA_PREDICCIONES).exists():
        log.info("predicciones.csv no existe → necesita entrenamiento inicial.")
        return True

    es_lunes = datetime.today().weekday() == 0
    if es_lunes:
        log.info("Es lunes → re-entrenamiento semanal programado.")
        return True

    # Contar filas nuevas desde la última predicción
    df_pred = pd.read_csv(RUTA_PREDICCIONES, parse_dates=["fecha"])
    ultima_pred = df_pred["fecha"].max()
    filas_nuevas = len(df[df["fecha"] > ultima_pred])

    if filas_nuevas > 30:
        log.info(f"{filas_nuevas} filas nuevas acumuladas → re-entrenamiento.")
        return True

    log.info(f"Re-entrenamiento no necesario ({filas_nuevas} filas nuevas desde última predicción).")
    return False


def reentrenar_y_exportar(df: pd.DataFrame):
    """
    Re-entrena Prophet y exporta predicciones.csv actualizado.
    Importa el pipeline de modelo_precios.py para no duplicar código.
    """
    log.info("Re-entrenando modelos Prophet (esto tarda ~5 minutos)...")
    try:
        # Guardar dataset actualizado temporalmente y correr el pipeline
        df.to_csv(RUTA_DATASET, index=False, encoding="utf-8")
        from modelo_precios import correr_pipeline
        correr_pipeline()
        log.info("✓ Modelos Prophet re-entrenados y predicciones exportadas.")
    except ImportError:
        log.error("No se encontró modelo_precios.py. Colocar en la misma carpeta.")
    except Exception as e:
        log.error(f"Error en re-entrenamiento: {e}")


# =============================================================================
# REGISTRO DE ACTUALIZACIONES
# =============================================================================

def registrar_actualizacion(n_filas: int, re_entrenado: bool):
    """Guarda un log simple de cada actualización para auditoría."""
    with open(RUTA_LOG_UPDATES, "a", encoding="utf-8") as f:
        f.write(
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
            f"Filas nuevas: {n_filas} | "
            f"Prophet re-entrenado: {'Sí' if re_entrenado else 'No'}\n"
        )


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def actualizar():
    """
    Pipeline completo de actualización diaria incremental.
    """
    log.info("=" * 55)
    log.info("PrecioJusto Campo — Actualización Diaria")
    log.info(f"Fecha: {datetime.today().strftime('%Y-%m-%d %H:%M')}")
    log.info("=" * 55)

    # ── 1. Cargar dataset existente ───────────────────────────────────────────
    if not Path(RUTA_DATASET).exists():
        log.error(f"No se encontró {RUTA_DATASET}. Ejecutar fase1_ingesta_granos.py primero.")
        return

    df = pd.read_csv(RUTA_DATASET, parse_dates=["fecha"])
    n_original = len(df)
    log.info(f"Dataset cargado: {n_original:,} filas | Hasta: {df['fecha'].max().date()}")

    # ── 2. Detectar días faltantes ────────────────────────────────────────────
    dias_faltantes = detectar_dias_faltantes(df)
    if not dias_faltantes:
        log.info("✅ Dataset ya está al día. Recalculando señales de todas formas...")
        df = recalcular_percentiles(df)
        df = recalcular_senales(df)
        df.to_csv(RUTA_DATASET, index=False, encoding="utf-8")
        return

    fecha_desde = min(dias_faltantes).date() if hasattr(min(dias_faltantes), 'date') else min(dias_faltantes)
    fecha_hasta = max(dias_faltantes).date() if hasattr(max(dias_faltantes), 'date') else max(dias_faltantes)

    # ── 3. Descargar solo los datos nuevos ────────────────────────────────────
    df_precios_nuevo = descargar_precios_nuevos(
        datetime.combine(fecha_desde, datetime.min.time()),
        datetime.combine(fecha_hasta, datetime.min.time()),
    )
    df_tc_nuevo = descargar_tc_nuevo(
        datetime.combine(fecha_desde, datetime.min.time()),
        datetime.combine(fecha_hasta, datetime.min.time()),
    )

    # ── 4. Integrar filas nuevas ──────────────────────────────────────────────
    df_actualizado = integrar_filas_nuevas(df, df_precios_nuevo, df_tc_nuevo)
    filas_agregadas = len(df_actualizado) - n_original

    # ── 5. Recalcular percentiles (siempre, es rápido) ────────────────────────
    df_actualizado = recalcular_percentiles(df_actualizado)

    # ── 6. Decidir si re-entrenar Prophet ─────────────────────────────────────
    re_entrenado = necesita_reentrenamiento(df_actualizado)
    if re_entrenado:
        df_actualizado.to_csv(RUTA_DATASET, index=False, encoding="utf-8")
        reentrenar_y_exportar(df_actualizado)
        # Recargar el dataset actualizado por el pipeline
        df_actualizado = pd.read_csv(RUTA_DATASET, parse_dates=["fecha"])
    else:
        # Solo recalcular señales con Prophet cacheado
        df_actualizado = recalcular_senales(df_actualizado)
        df_actualizado.to_csv(RUTA_DATASET, index=False, encoding="utf-8")
        log.info(f"✓ Dataset guardado: {RUTA_DATASET}")

    # ── 7. Registrar en log ───────────────────────────────────────────────────
    registrar_actualizacion(filas_agregadas, re_entrenado)

    # ── 8. Resumen ────────────────────────────────────────────────────────────
    log.info("")
    log.info("─" * 55)
    log.info("RESUMEN DE ACTUALIZACIÓN")
    log.info(f"  Filas agregadas   : {filas_agregadas}")
    log.info(f"  Dataset hasta     : {df_actualizado['fecha'].max().date()}")
    log.info(f"  Prophet re-entrenado: {'Sí' if re_entrenado else 'No (usando cache)'}")

    ultimo = df_actualizado[df_actualizado["percentil_soja"].notna()].iloc[-1] \
             if "percentil_soja" in df_actualizado.columns else None
    if ultimo is not None:
        for cultivo in ["soja", "maiz", "trigo"]:
            precio = ultimo.get(f"precio_{cultivo}", "N/A")
            perc   = ultimo.get(f"percentil_{cultivo}", "N/A")
            senal  = ultimo.get(f"senal_{cultivo}", "N/A")
            if not isinstance(precio, str):
                log.info(f"  {cultivo.upper():<6}: ${precio:.0f}/tn | P{perc:.0f} | {senal}")
    log.info("─" * 55)
    log.info("✅ Actualización completada. El dashboard ya refleja los datos nuevos.")


if __name__ == "__main__":
    actualizar()
