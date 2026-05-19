"""
=============================================================================
PrecioJusto Campo — Dashboard (Streamlit)
=============================================================================
Ejecutar con:
    streamlit run dashboard.py

Requiere: predicciones.csv y backtest_resultados.csv generados por modelo_precios.py
=============================================================================
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime, timedelta

def evaluar_estrategias_en_horizonte(
    df: pd.DataFrame,
    cultivo: str,
    horizonte_dias: int,
    umbral_percentil: int = 65,
    pct_suba: float = 5.0,
) -> dict:
    """
    Dado un horizonte de días, simula qué precio hubiera obtenido
    cada estrategia operando SOLO dentro de ventanas de ese tamaño
    a lo largo del histórico.

    Parámetros
    ----------
    df               : DataFrame con precio y percentil del cultivo
    cultivo          : "soja", "maiz" o "trigo"
    horizonte_dias   : días disponibles para vender (slider del usuario)
    umbral_percentil : umbral de señal VENDER para PrecioJusto
    pct_suba         : porcentaje de suba para estrategia +X%

    Retorna
    -------
    dict con resultados de cada estrategia y la recomendación final
    """
    col_precio = f"precio_{cultivo}"
    col_perc   = f"percentil_{cultivo}"

    df = df.dropna(subset=[col_precio]).copy()
    df = df[df["fecha"] <= pd.Timestamp.today()].copy()

    if len(df) < horizonte_dias:
        return {}

    precios    = df[col_precio].values
    fechas     = df["fecha"].values
    percentiles = df[col_perc].values if col_perc in df.columns else np.full(len(df), np.nan)

    resultados_pj    = []
    resultados_suba  = []
    resultados_cosecha = []
    resultados_forzado = []  # vender al final si no hubo señal

    # Deslizar ventana del tamaño del horizonte a lo largo del histórico
    paso = max(1, horizonte_dias // 4)  # cada cuarto de horizonte avanzamos

    for inicio in range(0, len(df) - horizonte_dias, paso):
        fin = inicio + horizonte_dias
        p_ventana    = precios[inicio:fin]
        f_ventana    = fechas[inicio:fin]
        perc_ventana = percentiles[inicio:fin]

        # ── PrecioJusto: primera señal VENDER en la ventana ──────────────────
        precio_pj = None
        for j, perc in enumerate(perc_ventana):
            if not np.isnan(perc) and perc >= umbral_percentil:
                precio_pj = p_ventana[j]
                break
        # Si no hay señal, vender al final (obligado)
        if precio_pj is None:
            precio_pj = p_ventana[-1]
            resultados_forzado.append(precio_pj)
        resultados_pj.append(precio_pj)

        # ── Estrategia +X% desde mínimo ──────────────────────────────────────
        precio_suba = None
        min_local   = p_ventana[0]
        for j in range(1, len(p_ventana)):
            if p_ventana[j] < min_local:
                min_local = p_ventana[j]
            elif (p_ventana[j] - min_local) / min_local * 100 >= pct_suba:
                precio_suba = p_ventana[j]
                break
        if precio_suba is None:
            precio_suba = p_ventana[-1]
        resultados_suba.append(precio_suba)

        # ── Venta forzada al final de la ventana ─────────────────────────────
        resultados_cosecha.append(p_ventana[-1])

    if not resultados_pj:
        return {}

    precio_pj_promedio   = np.mean(resultados_pj)
    precio_suba_promedio = np.mean(resultados_suba)
    precio_forzado_prom  = np.mean(resultados_cosecha)
    pct_forzadas_pj      = len(resultados_forzado) / len(resultados_pj) * 100

    # ── Determinar ganador ────────────────────────────────────────────────────
    if precio_pj_promedio >= precio_suba_promedio:
        ganador          = "PrecioJusto"
        ventaja          = precio_pj_promedio - precio_suba_promedio
        ventaja_pct      = ventaja / precio_suba_promedio * 100
    else:
        ganador          = f"Suba +{pct_suba:.0f}%"
        ventaja          = precio_suba_promedio - precio_pj_promedio
        ventaja_pct      = ventaja / precio_pj_promedio * 100

    # ── Generar explicación contextual ───────────────────────────────────────
    if horizonte_dias <= 15:
        contexto = "horizonte muy corto"
        explicacion = (
            f"Con solo {horizonte_dias} días disponibles, "
            f"hay poco margen para esperar señales. "
            f"{'PrecioJusto identifica si el precio actual ya es bueno históricamente.' if ganador == 'PrecioJusto' else f'La estrategia +{pct_suba:.0f}% puede capturar un rebote rápido si el mercado está activo.'}"
        )
    elif horizonte_dias <= 45:
        contexto = "horizonte corto"
        explicacion = (
            f"En {horizonte_dias} días "
            f"{'PrecioJusto suele encontrar al menos una ventana favorable.' if ganador == 'PrecioJusto' else f'la estrategia +{pct_suba:.0f}% históricamente captura mejor los rebotes en esta ventana.'} "
            f"{'El ' + str(round(pct_forzadas_pj)) + '% de las veces no hubo señal y se vendió al vencimiento.' if pct_forzadas_pj > 20 else ''}"
        )
    elif horizonte_dias <= 90:
        contexto = "horizonte medio"
        explicacion = (
            f"Con {horizonte_dias} días disponibles hay buen margen para operar. "
            f"{'PrecioJusto genera múltiples señales en este horizonte, permitiendo vender en partes.' if ganador == 'PrecioJusto' else f'La estrategia +{pct_suba:.0f}% captura bien las oscilaciones en este período.'}"
        )
    else:
        contexto = "horizonte largo"
        explicacion = (
            f"Con más de {horizonte_dias} días, "
            f"{'PrecioJusto tiene ventaja porque puede identificar el pico del ciclo estacional.' if ganador == 'PrecioJusto' else f'la estrategia +{pct_suba:.0f}% puede acumular varias señales de suba.'} "
            f"Considerar vender en partes para diversificar el riesgo de precio."
        )

    return {
        "horizonte_dias":        horizonte_dias,
        "contexto":              contexto,
        "precio_pj":             round(precio_pj_promedio, 2),
        "precio_suba":           round(precio_suba_promedio, 2),
        "precio_forzado":        round(precio_forzado_prom, 2),
        "ganador":               ganador,
        "ventaja_usd":           round(ventaja, 2),
        "ventaja_pct":           round(ventaja_pct, 2),
        "pct_sin_senal_pj":      round(pct_forzadas_pj, 1),
        "n_simulaciones":        len(resultados_pj),
        "explicacion":           explicacion,
    }


def generar_curva_horizontes(
    df: pd.DataFrame,
    cultivo: str,
    umbral_percentil: int = 65,
) -> pd.DataFrame:
    """
    Genera la curva completa de performance de cada estrategia
    para todos los horizontes de 10 a 180 días.
    Útil para el gráfico del dashboard.
    """
    horizontes = list(range(10, 181, 10))
    registros = []

    for h in horizontes:
        res = evaluar_estrategias_en_horizonte(df, cultivo, h, umbral_percentil)
        if res:
            registros.append({
                "horizonte":   h,
                "PrecioJusto": res["precio_pj"],
                f"Suba +5%":   res["precio_suba"],
                "ganador":     res["ganador"],
            })

    return pd.DataFrame(registros)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="PrecioJusto Campo",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS personalizado para look profesional
st.markdown("""
<style>
    .main { padding-top: 1rem; }
    .block-container { padding-top: 1.5rem; max-width: 1200px; }

    /* Tarjetas de métricas */
    [data-testid="metric-container"] {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 1rem;
    }

    /* Señal badge */
    .badge-verde  { background:#15ab39; color:#ffffff; padding:6px 16px; border-radius:20px; font-weight:600; font-size:15px; }
    .badge-amarillo { background:#879600; color:#ffffff; padding:6px 16px; border-radius:20px; font-weight:600; font-size:15px; }
    .badge-rojo   { background:#b5282a; color:#ffffff; padding:6px 16px; border-radius:20px; font-weight:600; font-size:15px; }

    /* Señal box */
    .signal-verde   { background:#000000; border-left:5px solid #28a745; padding:1rem 1.25rem; border-radius:6px; margin:1rem 0; }
    .signal-amarillo{ background:#000000; border-left:5px solid #ffc107; padding:1rem 1.25rem; border-radius:6px; margin:1rem 0; }
    .signal-rojo    { background:#000000; border-left:5px solid #dc3545; padding:1rem 1.25rem; border-radius:6px; margin:1rem 0; }

    /* Backtest card */
    .bt-card { background:#f0f4ff; border:1px solid #d0d9f5; border-radius:10px; padding:1rem 1.25rem; }
    .bt-number { font-size:28px; font-weight:700; color:#2d4699; }

    h1 { font-size: 1.6rem !important; }
    h2 { font-size: 1.2rem !important; }
    h3 { font-size: 1rem !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def cargar_datos() -> tuple:
    """Carga datos desde Google Drive o local según disponibilidad."""
    
    DRIVE_IDS = {
        "predicciones":       "1kO2xwUCcTYlg6a5yDepuyiqsLKZIBFAT",
        "backtest":           "1KjxmMX8gCqCzfuvQRauBeMjwaKkGhDRK",
    }

    def leer_csv_drive(file_id: str) -> pd.DataFrame:
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        try:
            return pd.read_csv(url)
        except Exception as e:
            st.error(f"Error leyendo desde Drive: {e}")
            return pd.DataFrame()

    # Intentar local primero, Drive como fallback
    if Path("predicciones.csv").exists():
        df = pd.read_csv("predicciones.csv", parse_dates=["fecha"])
        df_bt = pd.read_csv("backtest_resultados.csv") \
                if Path("backtest_resultados.csv").exists() else None
    else:
        st.info("Cargando datos desde la nube...")
        df = leer_csv_drive(DRIVE_IDS["predicciones"])
        df["fecha"] = pd.to_datetime(df["fecha"])
        df_bt = leer_csv_drive(DRIVE_IDS["backtest"])

    if df.empty:
        return None, None

    df = df.sort_values("fecha").reset_index(drop=True)
    return df, df_bt


df, df_bt = cargar_datos()

if df is None:
    st.error("⚠️ No se encontró predicciones.csv. Ejecutar primero: python modelo_precios.py")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🌾 PrecioJusto Campo")
    st.markdown("---")

    # Selector de cultivo
    cultivo = st.selectbox(
        "Cultivo",
        options=["soja", "maiz", "trigo"],
        format_func=lambda x: {"soja": "🟡 Soja", "maiz": "🟠 Maíz", "trigo": "🟤 Trigo"}[x],
    )

    # Selector de período
    periodo = st.select_slider(
        "Período histórico a mostrar",
        options=["1 mes","2 meses","6 meses", "1 año", "2 años", "5 años"],
        value="2 años",
    )

    meses_map = {"1 mes": 1,"2 meses": 2,"6 meses": 6, "1 año": 12, "2 años": 24, "5 años": 60}
    fecha_desde = datetime.today() - timedelta(days=meses_map[periodo] * 30)

    st.markdown("---")
    st.markdown("**Umbrales de señal**")
    umbral_vender  = st.slider("Percentil VENDER (≥)", 50, 90, 70)
    umbral_esperar = st.slider("Percentil ESPERAR (≤)", 10, 50, 40)

    st.markdown("---")
    st.caption(f"Actualizado: {df['fecha'].max().date()}")
    st.caption("Datos: CBOT · argentinadatos.com · MAGyP")


# ─────────────────────────────────────────────────────────────────────────────
# DATOS FILTRADOS
# ─────────────────────────────────────────────────────────────────────────────

col_precio    = f"precio_{cultivo}"
col_percentil = f"percentil_{cultivo}"
col_senal     = f"senal_{cultivo}"
col_pred      = f"pred_{cultivo}"
col_pred_lo   = f"pred_{cultivo}_lower"
col_pred_hi   = f"pred_{cultivo}_upper"
col_ars       = f"precio_{cultivo}_ars"

# Filtrar período
df_vista = df[df["fecha"] >= pd.Timestamp(fecha_desde)].copy()

# Último registro con datos de percentil
ultimo = df[df[col_percentil].notna()].iloc[-1]
precio_actual = ultimo[col_precio]
percentil_actual = ultimo[col_percentil]
tc_blue = ultimo.get("tipo_cambio_blue", 1)
precio_ars = precio_actual * tc_blue if tc_blue else None

# Recalcular señal en tiempo real con los umbrales actuales del sidebar
def recalcular_senal_live(percentil, pred_30d, precio, umbral_v, umbral_e):
    if pd.isna(percentil):
        return "N/A"
    # Si no hay predicción, decidir solo por percentil
    if pred_30d is None or pd.isna(pred_30d):
        if percentil >= umbral_v:
            return "VENDER"
        elif percentil <= umbral_e:
            return "ESPERAR"
        return "NEUTRAL"
    cambio_esp = ((pred_30d - precio) / precio * 100
                  if pred_30d and precio else 0)
    if percentil >= umbral_v:
        return "VENDER" if cambio_esp <= 5 else "NEUTRAL"
    elif percentil <= umbral_e:
        return "ESPERAR"
    return "NEUTRAL"
pred_30d_live = ultimo.get(f"pred_{cultivo}_30d")
COLOR_MAP = {"VENDER": "verde", "NEUTRAL": "amarillo", "ESPERAR": "rojo"}
senal_actual = recalcular_senal_live(
    percentil_actual, pred_30d_live, precio_actual,
    umbral_vender, umbral_esperar
)
color_actual = COLOR_MAP.get(senal_actual, "amarillo")
# Predicciones futuras
hoy = pd.Timestamp.today().normalize()
df_futuro = df[df["fecha"] > hoy]
pred_30d  = df_futuro.iloc[29][col_pred]  if len(df_futuro) > 29 else None
pred_60d  = df_futuro.iloc[59][col_pred]  if len(df_futuro) > 59 else None
pred_90d  = df_futuro.iloc[89][col_pred]  if len(df_futuro) > 89 else None




NOMBRE_CULTIVO = {"soja": "Soja", "maiz": "Maíz", "trigo": "Trigo"}[cultivo]


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.title(f"🌾 Monitor de Precios — {NOMBRE_CULTIVO}")
    st.caption(f"Análisis al {ultimo['fecha'].date()} · Modelo Prophet + Percentil histórico (3 años)")
with col_h2:
    st.markdown("<br>", unsafe_allow_html=True)
    emoji_senal = {"VENDER": "🟢", "NEUTRAL": "🟡", "ESPERAR": "🔴"}.get(senal_actual, "⚪")
    st.markdown(
        f'<div class="badge-{color_actual}">{emoji_senal} {senal_actual}</div>',
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────────────────────
# MÉTRICAS PRINCIPALES
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
m1, m2, m3, m4, m5 = st.columns(5)

with m1:
    st.metric(
        "Precio actual",
        f"${precio_actual:,.0f}",
        help="USD por tonelada métrica (CBOT)"
    )
with m2:
    st.metric(
        "Precio en ARS",
        f"${precio_ars:,.0f}" if precio_ars else "N/A",
        help="Convertido al tipo de cambio blue del día"
    )
with m3:
    delta_perc = percentil_actual - 50
    st.metric(
        "Percentil histórico",
        f"{percentil_actual:.0f}°",
        delta=f"{delta_perc:+.0f}° vs mediana",
        help="Posición del precio vs los últimos 3 años"
    )
with m4:
    if pred_30d:
        cambio_30 = (pred_30d - precio_actual) / precio_actual * 100
        st.metric(
            "Predicción 30 días",
            f"${pred_30d:,.0f}",
            delta=f"{cambio_30:+.1f}%",
            help="Proyección del modelo Prophet"
        )
with m5:
    st.metric(
        "TC Blue",
        f"${tc_blue:,.0f}" if tc_blue else "N/A",
        help="ARS por USD (tipo de cambio blue)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# SEÑAL PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

    TEXTOS_SENAL= {
    "VENDER": (
        "Momento favorable para vender",
        f"El precio de {NOMBRE_CULTIVO} está en el percentil {percentil_actual:.0f}° histórico "
        f"(últimos 3 años) y el modelo no proyecta una suba significativa en 30 días. "
        f"Precio de referencia en ARS: ${precio_ars:,.0f}/tn."
        if precio_ars else ""
    ),
    "NEUTRAL": (
        f"Zona intermedia — precio en percentil {percentil_actual:.0f}°",
        f"El precio de {NOMBRE_CULTIVO} está por encima del {umbral_esperar}° percentil "
        f"pero aún no alcanza el umbral de venta (P{umbral_vender}). "
        f"Faltan {umbral_vender - percentil_actual:.0f} puntos para señal VENDER. "
        f"Precio actual: ${precio_actual:,.0f} USD/tn — en ARS: ${precio_ars:,.0f}/tn."
        if precio_ars else
        f"El precio está {umbral_vender - percentil_actual:.0f} puntos por debajo del umbral de venta (P{umbral_vender})."
    ),
    
    "ESPERAR": (
        "Precio históricamente bajo — esperar recuperación",
        f"El precio está en el percentil {percentil_actual:.0f}° histórico, "
        "por debajo de la mediana de los últimos 3 años. "
        "El modelo sugiere aguardar una recuperación antes de vender."
    ),
}

titulo_senal, desc_senal = TEXTOS_SENAL.get(senal_actual, ("Sin datos suficientes", ""))
st.markdown(
    f'<div class="signal-{color_actual}"><strong>{titulo_senal}</strong><br>{desc_senal}</div>',
    unsafe_allow_html=True
)


# ─────────────────────────────────────────────────────────────────────────────
# GRÁFICO PRINCIPAL: PRECIO + PREDICCIÓN
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("### Precio histórico y predicción")

fig = go.Figure()

# Precio histórico
df_hist = df_vista[df_vista["fecha"] <= hoy].dropna(subset=[col_precio])
fig.add_trace(go.Scatter(
    x=df_hist["fecha"],
    y=df_hist[col_precio],
    mode="lines",
    name="Precio real",
    line=dict(color="#2d6a4f", width=2),
    hovertemplate="<b>%{x|%d/%m/%Y}</b><br>Precio: $%{y:,.0f}/tn<extra></extra>",
))

# Banda de predicción (intervalo de confianza)
if col_pred_lo in df.columns and col_pred_hi in df.columns:
    df_futuro_plot = df[df["fecha"] > hoy].dropna(subset=[col_pred])
    if not df_futuro_plot.empty:
        fig.add_trace(go.Scatter(
            x=pd.concat([df_futuro_plot["fecha"], df_futuro_plot["fecha"].iloc[::-1]]),
            y=pd.concat([df_futuro_plot[col_pred_hi], df_futuro_plot[col_pred_lo].iloc[::-1]]),
            fill="toself",
            fillcolor="rgba(55, 138, 221, 0.1)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Intervalo de confianza 80%",
            showlegend=True,
            hoverinfo="skip",
        ))

        # Línea de predicción
        fig.add_trace(go.Scatter(
            x=df_futuro_plot["fecha"],
            y=df_futuro_plot[col_pred],
            mode="lines",
            name="Predicción Prophet",
            line=dict(color="#185FA5", width=2, dash="dash"),
            hovertemplate="<b>%{x|%d/%m/%Y}</b><br>Predicción: $%{y:,.0f}/tn<extra></extra>",
        ))

# Línea vertical "hoy" — convertir a string ISO para compatibilidad con Plotly
fig.add_shape(
    type="line",
    x0=hoy,
    x1=hoy,
    y0=0,
    y1=1,
    xref="x",
    yref="paper",
    line=dict(color="gray", dash="dot")
)

fig.add_annotation(
    x=hoy,
    y=1,
    xref="x",
    yref="paper",
    text="Hoy",
    showarrow=False,
    yshift=10
)

# Señales VENDER en el período visible
if col_senal in df_vista.columns:
    df_ventas = df_vista[(df_vista[col_senal] == "VENDER") & (df_vista["fecha"] <= hoy)]
    if not df_ventas.empty:
        fig.add_trace(go.Scatter(
            x=df_ventas["fecha"],
            y=df_ventas[col_precio],
            mode="markers",
            name="Señal VENDER",
            marker=dict(color="#28a745", size=7, symbol="triangle-up"),
            hovertemplate="<b>Señal VENDER</b><br>%{x|%d/%m/%Y}<br>$%{y:,.0f}/tn<extra></extra>",
        ))

fig.update_layout(
    height=400,
    margin=dict(l=0, r=0, t=10, b=0),
    xaxis_title=None,
    yaxis_title="USD / tonelada",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="x unified",
    plot_bgcolor="white",
    paper_bgcolor="white",
    xaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
    yaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
)
st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# SEGUNDA FILA: PERCENTIL + PREDICCIONES
# ─────────────────────────────────────────────────────────────────────────────

col_perc, col_pred_tab = st.columns([3, 2])

with col_perc:
    st.markdown("### Percentil histórico rodante")

    df_perc = df_vista[df_vista["fecha"] <= hoy].dropna(subset=[col_percentil])

    fig_perc = go.Figure()
    fig_perc.add_hrect(y0=umbral_vender, y1=100, fillcolor="rgba(40,167,69,0.08)", line_width=0)
    fig_perc.add_hrect(y0=0, y1=umbral_esperar, fillcolor="rgba(220,53,69,0.08)", line_width=0)

    fig_perc.add_trace(go.Scatter(
        x=df_perc["fecha"],
        y=df_perc[col_percentil],
        mode="lines",
        name="Percentil",
        line=dict(color="#6f42c1", width=2),
        fill="tozeroy",
        fillcolor="rgba(111,66,193,0.05)",
        hovertemplate="<b>%{x|%d/%m/%Y}</b><br>Percentil: %{y:.0f}°<extra></extra>",
    ))

    # Líneas de umbral
    fig_perc.add_hline(y=umbral_vender, line_dash="dash", line_color="#28a745",
                       annotation_text=f"Umbral VENDER (P{umbral_vender})", annotation_position="right")
    fig_perc.add_hline(y=umbral_esperar, line_dash="dash", line_color="#dc3545",
                       annotation_text=f"Umbral ESPERAR (P{umbral_esperar})", annotation_position="right")

    fig_perc.update_layout(
        height=280,
        margin=dict(l=0, r=60, t=10, b=0),
        yaxis=dict(range=[0, 100], title="Percentil", showgrid=True, gridcolor="#f0f0f0"),
        xaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
    )
    st.plotly_chart(fig_perc, use_container_width=True)

with col_pred_tab:
    st.markdown("### Predicciones Prophet")
    st.markdown("<br>", unsafe_allow_html=True)

    pred_datos = [
        ("30 días", pred_30d),
        ("60 días", pred_60d),
        ("90 días", pred_90d),
    ]

    for label, pred_val in pred_datos:
        if pred_val:
            cambio = (pred_val - precio_actual) / precio_actual * 100
            color_delta = "#28a745" if cambio >= 0 else "#dc3545"
            flecha = "↑" if cambio >= 0 else "↓"
            st.markdown(
                f"""
                <div style="display:flex;justify-content:space-between;align-items:center;
                     padding:10px 0;border-bottom:1px solid #eee;">
                  <span style="color:#666;font-size:14px;">{label}</span>
                  <span style="font-size:16px;font-weight:600;">${pred_val:,.0f}/tn</span>
                  <span style="color:{color_delta};font-size:13px;font-weight:600;">
                    {flecha} {cambio:+.1f}%
                  </span>
                </div>
                """,
                unsafe_allow_html=True
            )

    # Precio en ARS proyectado
    if pred_30d and tc_blue:
        st.markdown("<br>", unsafe_allow_html=True)
        st.info(f"**Precio proyectado en ARS (30d):** ${pred_30d * tc_blue:,.0f} / tn")


# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("### Backtest de la estrategia (2022–2025)")
st.caption("¿Cuánto más hubiera cobrado un productor que siguió las señales vs vender en fecha aleatoria?")

if df_bt is not None and not df_bt.empty:
    cols_bt = st.columns(len(df_bt))
    for i, (_, row) in enumerate(df_bt.iterrows()):
        with cols_bt[i]:
            nombre = {"soja": "🟡 Soja", "maiz": "🟠 Maíz", "trigo": "🟤 Trigo"}.get(row["cultivo"], row["cultivo"])
            st.markdown(
                f"""
                <div class="bt-card">
                  <div style="font-size:13px;color:#666;margin-bottom:4px;">{nombre}</div>
                  <div class="bt-number">+{row['mejora_pct']:.1f}%</div>
                  <div style="font-size:12px;color:#555;margin-top:4px;">precio promedio superior</div>
                  <hr style="border:none;border-top:1px solid #dde;">
                  <div style="font-size:12px;color:#666;">
                    Modelo: <strong>${row['precio_modelo']:,.0f}/tn</strong><br>
                    Base: ${row['precio_promedio']:,.0f}/tn<br>
                    Señales emitidas: {row['n_senales']}
                  </div>
                </div>
                """,
                unsafe_allow_html=True
            )
else:
    st.info("Backtest no disponible. Verificar que backtest_resultados.csv existe.")


# ─────────────────────────────────────────────────────────────────────────────
# COMPARATIVO DE CULTIVOS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("### Comparativo de cultivos (hoy)")

cultivos_data = []
for c in ["soja", "maiz", "trigo"]:
    ult = df[df[f"percentil_{c}"].notna()].iloc[-1] if f"percentil_{c}" in df.columns else None
    if ult is not None:
        cultivos_data.append({
            "Cultivo": {"soja": "Soja", "maiz": "Maíz", "trigo": "Trigo"}[c],
            "Precio (USD/tn)": f"${ult[f'precio_{c}']:,.0f}",
            "Percentil": f"{ult[f'percentil_{c}']:.0f}°",
            "Señal": ult.get(f"senal_{c}", "N/A"),
            "Precio ARS/tn": f"${ult[f'precio_{c}'] * tc_blue:,.0f}" if tc_blue else "N/A",
        })

if cultivos_data:
    df_comp = pd.DataFrame(cultivos_data)

    def color_senal_celda(val):
        colores = {
            "VENDER":  "background-color: #d4edda; color: #155724; font-weight:bold",
            "NEUTRAL": "background-color: #fff3cd; color: #856404; font-weight:bold",
            "ESPERAR": "background-color: #f8d7da; color: #721c24; font-weight:bold",
        }
        return colores.get(val, "")

    styled = df_comp.style.map(color_senal_celda, subset=["Señal"])
    st.dataframe(styled, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# HISTORIAL DE SEÑALES
# ─────────────────────────────────────────────────────────────────────────────

with st.expander("Ver historial de señales recientes"):
    df_hist_senal = df[df["fecha"] <= hoy].dropna(subset=[col_senal, col_precio]).tail(60)
    df_hist_senal = df_hist_senal[["fecha", col_precio, col_percentil, col_senal]].copy()
    df_hist_senal.columns = ["Fecha", "Precio USD/tn", "Percentil", "Señal"]
    df_hist_senal["Fecha"] = df_hist_senal["Fecha"].dt.strftime("%d/%m/%Y")
    df_hist_senal["Precio USD/tn"] = df_hist_senal["Precio USD/tn"].map("${:,.0f}".format)
    df_hist_senal["Percentil"] = df_hist_senal["Percentil"].map("{:.0f}°".format)

    styled_hist = df_hist_senal.style.map(
        lambda v: {"VENDER": "background-color:#d4edda;color:#155724",
                   "NEUTRAL": "background-color:#fff3cd;color:#856404",
                   "ESPERAR": "background-color:#f8d7da;color:#721c24"}.get(v, ""),
        subset=["Señal"]
    )
    st.dataframe(styled_hist, use_container_width=True, hide_index=True, height=300)


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN: ESTRATEGIA ADAPTATIVA POR HORIZONTE
# Pegar en dashboard.py antes del footer
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("### ⏱️ ¿Cuándo conviene vender según tu horizonte?")
st.caption("Seleccioná cuántos días tenés disponibles para vender y el sistema recomienda la estrategia óptima")

col_hz1, col_hz2 = st.columns([2, 1])

with col_hz1:
    horizonte = st.slider(
        "Días disponibles para vender",
        min_value=7,
        max_value=180,
        value=60,
        step=7,
        format="%d días",
    )

with col_hz2:
    toneladas_hz = st.number_input(
        "Toneladas a vender",
        min_value=100,
        max_value=100000,
        value=500,
        step=100,
    )

# Evaluar estrategias para el horizonte seleccionado
fecha_corte_hz = pd.Timestamp.today() - pd.DateOffset(months=18)
df_hz = df[(df["fecha"] >= fecha_corte_hz) & (df["fecha"] <= pd.Timestamp.today())]

res = evaluar_estrategias_en_horizonte(
    df_hz,
    cultivo,
    horizonte_dias=horizonte,
    umbral_percentil=umbral_vender,
)

if res:
    # ── Métricas ──────────────────────────────────────────────────────────
    mh1, mh2, mh3, mh4 = st.columns(4)
    with mh1:
        st.metric(
            "Estrategia recomendada",
            res["ganador"],
            help="Basado en performance histórica en ventanas similares"
        )
    with mh2:
        st.metric(
            "Precio esperado PrecioJusto",
            f"${res['precio_pj']:,.0f}/tn",
        )
    with mh3:
        st.metric(
            "Precio esperado Suba +5%",
            f"${res['precio_suba']:,.0f}/tn",
        )
    with mh4:
        diferencia_total = (res["precio_pj"] - res["precio_suba"]) * toneladas_hz
        st.metric(
            "Diferencia total",
            f"${abs(diferencia_total):,.0f} USD",
            delta=f"{'PrecioJusto' if diferencia_total > 0 else 'Suba +5%'} gana",
        )

    # ── Señal de recomendación ────────────────────────────────────────────
    color_hz = "verde" if res["ganador"] == "PrecioJusto" else "amarillo"
    bg_hz    = {"verde": "#d4edda", "amarillo": "#fff3cd"}
    borde_hz = {"verde": "#28a745", "amarillo": "#ffc107"}
    emoji_hz = "🤖" if res["ganador"] == "PrecioJusto" else "📈"

    advertencia_sin_senal = ""
    if res["pct_sin_senal_pj"] > 15:
        advertencia_sin_senal = (
            f'<br><span style="font-size:12px;color:#856404;">'
            f'⚠️ En el {res["pct_sin_senal_pj"]:.0f}% de las ventanas históricas '
            f'PrecioJusto no emitió señal a tiempo y se vendió al vencimiento.</span>'
        )

    st.markdown(
        f"""
        <div style="background:{bg_hz[color_hz]};
             border-left:4px solid {borde_hz[color_hz]};
             padding:1rem 1.25rem; border-radius:6px; margin-top:0.5rem;">
          <strong>{emoji_hz} En un horizonte de {horizonte} días ({res["contexto"]}),
          la estrategia recomendada es {res["ganador"]}</strong>
          con una ventaja histórica de ${res["ventaja_usd"]:.0f}/tn (+{res["ventaja_pct"]:.1f}%).<br>
          <span style="font-size:13px;">{res["explicacion"]}</span>
          {advertencia_sin_senal}
        </div>
        """,
        unsafe_allow_html=True
    )

    # ── Gráfico: curva de performance por horizonte ───────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Performance histórica de cada estrategia según horizonte disponible**")
    st.caption("Precio promedio obtenido simulando cada estrategia en ventanas históricas del mismo tamaño")

    fecha_corte_hz = pd.Timestamp.today() - pd.DateOffset(months=18)
df_hz = df[(df["fecha"] >= fecha_corte_hz) & (df["fecha"] <= pd.Timestamp.today())]

with st.spinner("Calculando curva de horizontes..."):
    df_curva = generar_curva_horizontes(
        df_hz,
        cultivo,
        umbral_percentil=umbral_vender,
    )

    if not df_curva.empty:
        fig_hz = go.Figure()

        fig_hz.add_trace(go.Scatter(
            x=df_curva["horizonte"],
            y=df_curva["PrecioJusto"],
            mode="lines+markers",
            name="PrecioJusto",
            line=dict(color="#28a745", width=2),
            marker=dict(size=6),
            hovertemplate="<b>%{x} días</b><br>PrecioJusto: $%{y:,.0f}/tn<extra></extra>",
        ))

        fig_hz.add_trace(go.Scatter(
            x=df_curva["horizonte"],
            y=df_curva["Suba +5%"],
            mode="lines+markers",
            name="Suba +5%",
            line=dict(color="#e67e22", width=2, dash="dash"),
            marker=dict(size=6),
            hovertemplate="<b>%{x} días</b><br>Suba +5%%: $%{y:,.0f}/tn<extra></extra>",
        ))

        # Marcar el horizonte seleccionado
        fig_hz.add_shape(
            type="line",
            x0=horizonte, x1=horizonte,
            y0=0, y1=1,
            xref="x", yref="paper",
            line=dict(color="gray", width=1, dash="dot"),
        )
        fig_hz.add_annotation(
            x=horizonte, y=1, yref="paper",
            text=f"Tu horizonte ({horizonte}d)",
            showarrow=False,
            yanchor="bottom",
            font=dict(size=11, color="gray"),
        )

        fig_hz.update_layout(
            height=280,
            margin=dict(l=0, r=0, t=20, b=0),
            xaxis_title="Días disponibles para vender",
            yaxis_title="Precio promedio obtenido (USD/tn)",
            legend=dict(orientation="h", y=1.1),
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
            yaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
        )
        st.plotly_chart(fig_hz, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────


st.markdown("---")
st.caption(
    "PrecioJusto Campo · Datos: Yahoo Finance (CBOT) · argentinadatos.com · MAGyP Argentina · "
    "Modelo: Prophet (Meta) · Este análisis es orientativo y no constituye asesoramiento financiero."
)
