# Agrosignal


Sistema de análisis y predicción de precios de granos para productores agropecuarios argentinos. Integra datos de mercados internacionales (CBOT), tipo de cambio, fuentes argentinas (ROFEX, CIARA-CEC) y modelos de machine learning para emitir señales de venta sobre soja, maíz y trigo.

---

## Tabla de contenidos

- [Descripción general](#descripción-general)
- [Arquitectura del sistema](#arquitectura-del-sistema)
- [Tecnologías utilizadas](#tecnologías-utilizadas)
- [Requisitos previos](#requisitos-previos)
- [Instalación](#instalación)
- [Uso](#uso)
- [Módulos del proyecto](#módulos-del-proyecto)
- [Fuentes de datos](#fuentes-de-datos)
- [Modelo de machine learning](#modelo-de-machine-learning)
- [Señales de precio](#señales-de-precio)
- [Decisiones técnicas](#decisiones-técnicas)

---

## Descripción general

PrecioJusto Campo responde una pregunta concreta del productor: **¿es hoy un buen momento para vender?**

Para responderla, el sistema:

1. Descarga precios históricos de soja, maíz y trigo desde mercados internacionales.
2. Incorpora el tipo de cambio blue y datos de retención de cosecha (CIARA-CEC).
3. Calcula en qué percentil histórico se encuentra el precio de hoy (últimos 12 meses).
4. Entrena un modelo Prophet que predice precios a 30, 60 y 90 días.
5. Combina ambas señales para emitir un semáforo: **VENDER / NEUTRAL / ESPERAR**.
6. Muestra todo en un dashboard interactivo con backtest histórico (2022–2025).

---

## Arquitectura del sistema

```
fase1_ingesta_granos.py
        │
        ▼
dataset_maestro_granos.csv
        │
        ├── patch_stocks_mundiales.py  (datos USDA embebidos)
        │
        ▼
modelo_precios.py
        │
        ▼
predicciones.csv + backtest_resultados.csv
        │
        ├── fuentes_arg.py  (ROFEX + CIARA-CEC)
        │
        ▼
dashboard.py  (Streamlit)
        │
        ▼
actualizar_diario.py  (actualización incremental)
```

Los módulos se encadenan mediante archivos CSV intermedios. El dashboard consume `predicciones.csv` y puede regenerarse sin re-correr la ingesta completa.

---

## Tecnologías utilizadas

| Capa | Tecnología |
|------|-----------|
| Dashboard | Streamlit, Plotly |
| ML / predicción | Prophet (Meta), scikit-learn |
| Datos numéricos | pandas, NumPy |
| Scraping / APIs | requests, BeautifulSoup4, yfinance |
| Scheduler | schedule |
| Descarga de archivos | gdown |

---

## Requisitos previos

- Python 3.9 o superior
- pip

---

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/NicoGarcia25/AppCampo.git
cd AppCampo

# 2. Crear y activar entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Instalar dependencias
pip install -r requirements.txt
```

---

## Uso

El pipeline se corre en tres pasos, en orden:

### Paso 1 — Ingesta de datos

```bash
python fase1_ingesta_granos.py
```

Descarga precios CBOT, tipo de cambio y producción argentina. Genera `dataset_maestro_granos.csv`.

```bash
python patch_stocks_mundiales.py
```

Agrega los datos de stocks mundiales (USDA WASDE) al dataset maestro.

### Paso 2 — Modelo de precios

```bash
python modelo_precios.py
```

Calcula percentiles históricos, entrena Prophet y genera señales. Produce `predicciones.csv` y `backtest_resultados.csv`.

### Paso 3 — Dashboard

```bash
streamlit run dashboard.py
```

Abre el dashboard en `http://localhost:8501`.

### Actualización diaria (opcional)

```bash
python actualizar_diario.py
```

Descarga solo los días faltantes sin re-correr la ingesta completa. Re-entrena Prophet los lunes o cuando se acumulan más de 30 días nuevos.

---

## Módulos del proyecto

| Archivo | Responsabilidad |
|---------|----------------|
| `fase1_ingesta_granos.py` | Descarga y limpieza de precios CBOT, tipo de cambio y producción ARG |
| `patch_stocks_mundiales.py` | Embebe datos de stocks mundiales USDA WASDE (2019–2024) |
| `modelo_precios.py` | Percentil rodante, Prophet, señales VENDER/NEUTRAL/ESPERAR, backtest |
| `fuentes_arg.py` | Datos de dólar futuro ROFEX y liquidaciones CIARA-CEC |
| `estrategia_adaptativa.py` | Evaluación de estrategias según horizonte de venta del productor |
| `actualizar_diario.py` | Actualización incremental del dataset y las predicciones |
| `dashboard.py` | Interfaz Streamlit con visualizaciones y señales en tiempo real |

---

## Fuentes de datos

| Dato | Fuente | Método |
|------|--------|--------|
| Precios CBOT (soja, maíz, trigo) | Yahoo Finance (`ZS=F`, `ZC=F`, `ZW=F`) | yfinance + fallback Stooq |
| Tipo de cambio blue | argentinadatos.com | API pública |
| Producción ARG | datos.gob.ar (MAGyP) | CSV directo |
| Stocks mundiales | USDA WASDE | Datos embebidos + FAOSTAT API |
| Dólar futuro | ROFEX / rava.com | Scraping |
| Retención de cosecha | CIARA-CEC | Web scraping |

Todas las fuentes son públicas y no requieren autenticación.

---

## Modelo de machine learning

### Percentil histórico rodante

Para cada día, el sistema calcula en qué percentil se encuentra el precio actual respecto a los últimos 365 días:

- **Percentil > 65** → el precio de hoy es mayor al 65% de los precios del último año → señal de venta favorable.
- **Percentil < 40** → precio bajo históricamente → conviene esperar.

### Predicción Prophet

Se entrena un modelo [Prophet](https://facebook.github.io/prophet/) por cultivo con precios históricos diarios. El modelo genera predicciones a 30, 60 y 90 días con intervalos de confianza.

### Backtest (2022–2025)

El sistema simula cuánto hubiera ganado un productor que vendió en cada señal VENDER vs. vender en un día aleatorio. Los resultados se exportan a `backtest_resultados.csv` y se muestran en el dashboard.

---

## Señales de precio

| Señal | Condición | Color |
|-------|-----------|-------|
| **VENDER** | Percentil ≥ 65 y predicción a 30d no supera el precio actual significativamente | 🟢 Verde |
| **NEUTRAL** | Zona intermedia | 🟡 Amarillo |
| **ESPERAR** | Percentil ≤ 40 o predicción indica suba | 🔴 Rojo |

La **estrategia adaptativa** ajusta la recomendación según el horizonte de días disponibles que ingresa el productor en el dashboard.

---

## Decisiones técnicas

**¿Por qué Streamlit y no Flask/Django?**
El foco del proyecto es el análisis y la visualización de datos. Streamlit permite construir dashboards interactivos con Python puro, sin necesidad de HTML/CSS adicional ni un backend separado.

**¿Por qué Prophet y no LSTM/ARIMA?**
Prophet maneja bien series temporales con estacionalidad y datos faltantes (feriados, fines de semana sin mercado), con configuración mínima y resultados interpretables. Para un MVP de predicción de commodities agrícolas es más robusto que modelos ARIMA manuales y más fácil de auditar que una red neuronal.

**¿Por qué percentil rodante y no precio absoluto?**
El precio absoluto no dice si hoy es "caro" o "barato". El percentil contextualiza el precio en su historia reciente, que es lo que un productor necesita para decidir cuándo vender.

**¿Por qué archivos CSV entre módulos y no una base de datos?**
Para un proyecto de análisis con datos históricos estáticos (5 años de precios diarios ≈ 1800 filas), los CSVs son suficientes, portables y fáciles de auditar. Una base de datos agregaría complejidad operacional sin beneficio real en este volumen.

**¿Por qué múltiples fuentes con fallback?**
Las APIs agrícolas argentinas son inestables (datos.gob.ar, MAGyP, ROFEX). El diseño con fuentes primarias y fallbacks garantiza que la ingesta funcione aunque alguna fuente esté caída.
