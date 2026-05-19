"""
=============================================================================
PrecioJusto Campo — Estrategia Adaptativa por Horizonte Temporal
=============================================================================
Dado un horizonte de días para vender, el sistema evalúa qué estrategia
históricamente hubiera dado mejor precio en ventanas similares y recomienda
la más adecuada con explicación.

Para integrar al dashboard: ver instrucciones al final del archivo.
=============================================================================
"""

import pandas as pd
import numpy as np


# =============================================================================
# MOTOR DE ESTRATEGIA ADAPTATIVA
# =============================================================================

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


# =============================================================================
# BLOQUE PARA PEGAR EN dashboard.py
# =============================================================================
# Instrucciones:
#   1. Copiá la función evaluar_estrategias_en_horizonte() y
#      generar_curva_horizontes() al inicio de dashboard.py
#      (después de los imports)
#   2. Pegá el bloque de UI de abajo antes del footer del dashboard
# =============================================================================

DASHBOARD_BLOCK = '''
# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN: ESTRATEGIA ADAPTATIVA POR HORIZONTE
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
res = evaluar_estrategias_en_horizonte(
    df[df["fecha"] <= pd.Timestamp.today()],
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
            f"${res[\'precio_pj\']:,.0f}/tn",
        )
    with mh3:
        st.metric(
            f"Precio esperado Suba +5%",
            f"${res[\'precio_suba\']:,.0f}/tn",
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

    st.markdown(
        f"""
        <div style="background:{bg_hz[color_hz]};
             border-left:4px solid {borde_hz[color_hz]};
             padding:1rem 1.25rem; border-radius:6px; margin-top:0.5rem;">
          <strong>{emoji_hz} En un horizonte de {horizonte} días ({res["contexto"]}),
          la estrategia recomendada es {res["ganador"]}</strong>
          con una ventaja histórica de ${res["ventaja_usd"]:.0f}/tn (+{res["ventaja_pct"]:.1f}%).<br>
          <span style="font-size:13px;">{res["explicacion"]}</span>
          {f\'<br><span style="font-size:12px;color:#856404;">⚠️ En el {res["pct_sin_senal_pj"]:.0f}% de las ventanas históricas PrecioJusto no emitió señal a tiempo y se vendió al vencimiento.</span>\'
           if res["pct_sin_senal_pj"] > 15 else ""}
        </div>
        """,
        unsafe_allow_html=True
    )

    # ── Gráfico: curva de performance por horizonte ───────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Performance histórica de cada estrategia según horizonte disponible**")
    st.caption("Precio promedio obtenido simulando cada estrategia en ventanas históricas del mismo tamaño")

    with st.spinner("Calculando curva de horizontes..."):
        df_curva = generar_curva_horizontes(
            df[df["fecha"] <= pd.Timestamp.today()],
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
        fig_hz.add_vline(
            x=horizonte,
            line_dash="dot",
            line_color="gray",
            annotation_text=f"Tu horizonte ({horizonte}d)",
            annotation_position="top",
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
'''
