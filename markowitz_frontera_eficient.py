
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf

warnings.filterwarnings("ignore")

# =============================================================================
# BLOQUE 1: EXTRACCIÓN DE DATOS (Yahoo Finance)
# =============================================================================

TICKERS = [
    "AAPL", "MSFT", "NVDA",        # NASDAQ (EE. UU.)
    "BA", "GS", "DIS",             # US30 / Dow Jones (EE. UU.)
    "AZN.L", "HSBA.L", "ULVR.L",   # FTSE 100 (Reino Unido)
    "SIE.DE", "ALV.DE", "VOW3.DE", # DAX (Alemania)
    "7203.T", "6758.T", "9984.T"   # Nikkei 225 (Japón)
]

OUTPUT_DIR = Path.home() / "Desktop" / "markowitz_resultados"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("  FRONTERA EFICIENTE DE MARKOWITZ - SIMULACIÓN DE MONTE CARLO")
print("=" * 65)
print(f"\n[1/5] Descargando 10 años de datos históricos para {len(TICKERS)} activos...")

try:
    datos_crudos = yf.download(TICKERS, period="10y", auto_adjust=False, progress=False)
    precios = datos_crudos["Adj Close"].dropna()
except Exception as e:
    raise RuntimeError(
        "No se pudieron descargar datos desde Yahoo Finance. "
        "Verifica conexión a internet y disponibilidad de los tickers."
    ) from e

if precios.empty:
    raise RuntimeError(
        "La descarga no devolvió precios válidos. "
        "Revisa los tickers o intenta nuevamente más tarde."
    )

print(f"      Datos descargados: {precios.shape[0]} días de trading, {precios.shape[1]} activos.")
print(f"      Período: {precios.index[0].strftime('%Y-%m-%d')} → {precios.index[-1].strftime('%Y-%m-%d')}")
print(f"      Activos disponibles: {list(precios.columns)}")

# =============================================================================
# BLOQUE 2: ANÁLISIS INICIAL Y MATRIZ DE CORRELACIÓN
# =============================================================================

print("\n[2/5] Calculando retornos diarios y generando matriz de correlación...")

retornos_diarios = precios.pct_change().dropna()
DIAS_TRADING = 252

retorno_medio_diario = retornos_diarios.mean()
matriz_cov_anualizada = retornos_diarios.cov() * DIAS_TRADING

print(f"      Retornos diarios calculados: {retornos_diarios.shape[0]} observaciones.")

matriz_correlacion = retornos_diarios.corr()

fig1, ax1 = plt.subplots(figsize=(12, 9))
sns.heatmap(
    matriz_correlacion,
    annot=True,
    fmt=".2f",
    cmap="RdYlGn",
    center=0,
    vmin=-1,
    vmax=1,
    linewidths=0.5,
    linecolor="white",
    ax=ax1,
    cbar_kws={"shrink": 0.8, "label": "Coeficiente de Correlación"},
)
ax1.set_title(
    "Matriz de Correlación de Retornos Diarios\n"
    "(15 Activos Internacionales Diversificados)",
    fontsize=14,
    fontweight="bold",
    pad=15,
)
ax1.tick_params(axis="x", rotation=45, labelsize=9)
ax1.tick_params(axis="y", rotation=0, labelsize=9)
plt.tight_layout()

correlacion_png = OUTPUT_DIR / "correlacion_activos.png"
fig1.savefig(correlacion_png, dpi=150, bbox_inches="tight")
plt.close(fig1)

print(f"      Matriz de correlación guardada en: {correlacion_png}")

# =============================================================================
# BLOQUE 3: SIMULACIÓN DE MONTE CARLO
# =============================================================================

print("\n[3/5] Ejecutando Simulación de Monte Carlo (100,000 portafolios)...")

NUM_PORTAFOLIOS = 100_000
NUM_ACTIVOS = len(precios.columns)
# Supuesto para Sharpe Ratio: tasa libre de riesgo anual de referencia.
# Ejemplo de referencia: bono del Tesoro de EE. UU. a 10 años (abril 2026, ~4.2%).
# Actualizar este valor según mercado y fecha del análisis.
TASA_LIBRE_RIESGO = 0.042
RANDOM_SEED = 42
MIN_PESO = 0.001

resultados = np.zeros((NUM_PORTAFOLIOS, 3))
pesos_sim = np.zeros((NUM_PORTAFOLIOS, NUM_ACTIVOS))

retorno_medio_anual = retorno_medio_diario.values * DIAS_TRADING

np.random.seed(RANDOM_SEED)

for i in range(NUM_PORTAFOLIOS):
    pesos_aleatorios = np.random.random(NUM_ACTIVOS)
    pesos_aleatorios /= pesos_aleatorios.sum()

    rendimiento = np.dot(pesos_aleatorios, retorno_medio_anual)
    varianza = pesos_aleatorios @ matriz_cov_anualizada.values @ pesos_aleatorios
    volatilidad = np.sqrt(varianza)
    sharpe = (rendimiento - TASA_LIBRE_RIESGO) / volatilidad

    resultados[i, 0] = rendimiento
    resultados[i, 1] = volatilidad
    resultados[i, 2] = sharpe
    pesos_sim[i, :] = pesos_aleatorios

print(f"      Simulación completada. {NUM_PORTAFOLIOS:,} portafolios evaluados.")

df_portafolios = pd.DataFrame(resultados, columns=["Rendimiento", "Volatilidad", "Sharpe"])

# =============================================================================
# BLOQUE 4: OPTIMIZACIÓN — PORTAFOLIOS CRÍTICOS
# =============================================================================

print("\n[4/5] Identificando portafolios óptimos...")

idx_max_sharpe = df_portafolios["Sharpe"].idxmax()
portafolio_max_sharpe = df_portafolios.loc[idx_max_sharpe]
pesos_max_sharpe = pesos_sim[idx_max_sharpe]

idx_min_var = df_portafolios["Volatilidad"].idxmin()
portafolio_min_var = df_portafolios.loc[idx_min_var]
pesos_min_var = pesos_sim[idx_min_var]


def print_metric_row(label: str, value: str) -> None:
    print(f"  │  {label:<30} {value:>23}  │")


def print_weight_row(ticker: str, peso: float) -> None:
    percentage_str = f"{peso*100:.2f}%"
    print(f"  │    {ticker:<8}: {percentage_str:>8}                                      │")

print("\n  ┌─────────────────────────────────────────────────────────────┐")
print("  │           PORTAFOLIO DE MÁXIMO RATIO DE SHARPE              │")
print("  ├─────────────────────────────────────────────────────────────┤")
print_metric_row("Rendimiento Esperado (E[R_p]):", f"{portafolio_max_sharpe['Rendimiento']*100:.2f}%")
print_metric_row("Volatilidad Anualizada (σ_p):", f"{portafolio_max_sharpe['Volatilidad']*100:.2f}%")
print_metric_row("Ratio de Sharpe:", f"{portafolio_max_sharpe['Sharpe']:.4f}")
print("  ├─────────────────────────────────────────────────────────────┤")
print("  │  Pesos del portafolio:                                      │")
for ticker, peso in zip(precios.columns, pesos_max_sharpe):
    if peso > MIN_PESO:
        print_weight_row(ticker, peso)
print("  └─────────────────────────────────────────────────────────────┘")

print("\n  ┌─────────────────────────────────────────────────────────────┐")
print("  │        PORTAFOLIO DE MÍNIMA VARIANZA GLOBAL (GMV)          │")
print("  ├─────────────────────────────────────────────────────────────┤")
print_metric_row("Rendimiento Esperado (E[R_p]):", f"{portafolio_min_var['Rendimiento']*100:.2f}%")
print_metric_row("Volatilidad Anualizada (σ_p):", f"{portafolio_min_var['Volatilidad']*100:.2f}%")
print_metric_row("Ratio de Sharpe:", f"{portafolio_min_var['Sharpe']:.4f}")
print("  ├─────────────────────────────────────────────────────────────┤")
print("  │  Pesos del portafolio:                                      │")
for ticker, peso in zip(precios.columns, pesos_min_var):
    if peso > MIN_PESO:
        print_weight_row(ticker, peso)
print("  └─────────────────────────────────────────────────────────────┘")

df_pesos_max_sharpe = pd.DataFrame(
    {"Activo": precios.columns, "Peso (%)": (pesos_max_sharpe * 100).round(2)}
).sort_values("Peso (%)", ascending=False).reset_index(drop=True)

df_pesos_min_var = pd.DataFrame(
    {"Activo": precios.columns, "Peso (%)": (pesos_min_var * 100).round(2)}
).sort_values("Peso (%)", ascending=False).reset_index(drop=True)

# Exportación de resultados a archivos en Escritorio
pesos_max_csv = OUTPUT_DIR / "pesos_max_sharpe.csv"
pesos_gmv_csv = OUTPUT_DIR / "pesos_gmv.csv"
resumen_txt = OUTPUT_DIR / "resumen_markowitz.txt"


df_pesos_max_sharpe.to_csv(pesos_max_csv, index=False, encoding="utf-8")
df_pesos_min_var.to_csv(pesos_gmv_csv, index=False, encoding="utf-8")

with resumen_txt.open("w", encoding="utf-8") as f:
    f.write("=" * 65 + "\n")
    f.write("ANÁLISIS COMPLETADO\n")
    f.write("=" * 65 + "\n")
    f.write(f"Activos analizados: {NUM_ACTIVOS}\n")
    f.write(f"Portafolios simulados: {NUM_PORTAFOLIOS:,}\n")
    f.write(f"Tasa libre de riesgo (R_f): {TASA_LIBRE_RIESGO*100:.1f}%\n")
    f.write(f"Días de trading anualizados: {DIAS_TRADING}\n\n")

    f.write("Portafolio Max Sharpe:\n")
    f.write(f"  Rendimiento: {portafolio_max_sharpe['Rendimiento']*100:.2f}%\n")
    f.write(f"  Volatilidad: {portafolio_max_sharpe['Volatilidad']*100:.2f}%\n")
    f.write(f"  Sharpe Ratio: {portafolio_max_sharpe['Sharpe']:.4f}\n\n")

    f.write("Portafolio GMV (Mínima Varianza):\n")
    f.write(f"  Rendimiento: {portafolio_min_var['Rendimiento']*100:.2f}%\n")
    f.write(f"  Volatilidad: {portafolio_min_var['Volatilidad']*100:.2f}%\n")
    f.write(f"  Sharpe Ratio: {portafolio_min_var['Sharpe']:.4f}\n\n")

    f.write("Tabla de pesos — Portafolio Max Sharpe:\n")
    f.write(df_pesos_max_sharpe.to_string(index=False) + "\n\n")

    f.write("Tabla de pesos — Portafolio GMV:\n")
    f.write(df_pesos_min_var.to_string(index=False) + "\n")

print(f"      Pesos Max Sharpe guardados en: {pesos_max_csv}")
print(f"      Pesos GMV guardados en: {pesos_gmv_csv}")
print(f"      Resumen guardado en: {resumen_txt}")

# =============================================================================
# BLOQUE 5: VISUALIZACIÓN AVANZADA — FRONTERA EFICIENTE
# =============================================================================

print("\n[5/5] Generando visualización de la Frontera Eficiente...")

fig2, ax2 = plt.subplots(figsize=(14, 9))

scatter = ax2.scatter(
    df_portafolios["Volatilidad"] * 100,
    df_portafolios["Rendimiento"] * 100,
    c=df_portafolios["Sharpe"],
    cmap="viridis",
    alpha=0.4,
    s=5,
    zorder=1,
    label="_nolegend_",
)

cbar = plt.colorbar(scatter, ax=ax2, pad=0.02)
cbar.set_label("Ratio de Sharpe", fontsize=12, labelpad=10)
cbar.ax.tick_params(labelsize=10)

ax2.scatter(
    portafolio_max_sharpe["Volatilidad"] * 100,
    portafolio_max_sharpe["Rendimiento"] * 100,
    marker="*",
    s=600,
    color="gold",
    edgecolors="black",
    linewidths=1.5,
    zorder=5,
    label=(
        f"Max Sharpe  |  σ={portafolio_max_sharpe['Volatilidad']*100:.2f}%"
        f"  |  E[R]={portafolio_max_sharpe['Rendimiento']*100:.2f}%"
        f"  |  SR={portafolio_max_sharpe['Sharpe']:.4f}"
    ),
)

ax2.scatter(
    portafolio_min_var["Volatilidad"] * 100,
    portafolio_min_var["Rendimiento"] * 100,
    marker="*",
    s=600,
    color="red",
    edgecolors="black",
    linewidths=1.5,
    zorder=5,
    label=(
        f"Min Varianza (GMV)  |  σ={portafolio_min_var['Volatilidad']*100:.2f}%"
        f"  |  E[R]={portafolio_min_var['Rendimiento']*100:.2f}%"
        f"  |  SR={portafolio_min_var['Sharpe']:.4f}"
    ),
)

ax2.annotate(
    "Max Sharpe",
    xy=(portafolio_max_sharpe["Volatilidad"] * 100, portafolio_max_sharpe["Rendimiento"] * 100),
    xytext=(8, 5),
    textcoords="offset points",
    fontsize=10,
    fontweight="bold",
    color="black",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="gold", alpha=0.7),
)

ax2.annotate(
    "GMV",
    xy=(portafolio_min_var["Volatilidad"] * 100, portafolio_min_var["Rendimiento"] * 100),
    xytext=(8, -15),
    textcoords="offset points",
    fontsize=10,
    fontweight="bold",
    color="white",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.8),
)

ax2.set_title(
    "Frontera Eficiente de Markowitz\n"
    f"Simulación de Monte Carlo — {NUM_PORTAFOLIOS:,} Portafolios | "
    f"Risk-Free Rate: {TASA_LIBRE_RIESGO*100:.1f}% | "
    f"{len(TICKERS)} Activos Internacionales",
    fontsize=13,
    fontweight="bold",
    pad=15,
)
ax2.set_xlabel("Riesgo — Volatilidad Anualizada (%)", fontsize=12, labelpad=10)
ax2.set_ylabel("Retorno Esperado Anualizado (%)", fontsize=12, labelpad=10)
ax2.legend(loc="upper left", fontsize=9, framealpha=0.9, edgecolor="gray", fancybox=True)
ax2.grid(True, linestyle="--", alpha=0.5, color="gray")
ax2.set_facecolor("#f8f9fa")
fig2.patch.set_facecolor("white")

plt.tight_layout()
frontera_png = OUTPUT_DIR / "frontera_eficiente_markowitz.png"
fig2.savefig(frontera_png, dpi=150, bbox_inches="tight")
plt.close(fig2)

print(f"      Frontera Eficiente guardada en: {frontera_png}")

print("\n" + "=" * 65)
print("  ANÁLISIS COMPLETADO")
print("=" * 65)
print(f"  • Archivos exportados en: {OUTPUT_DIR}")
print("=" * 65)
