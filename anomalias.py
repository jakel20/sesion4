import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Detector de Anomalías — Módulo 4", layout="wide")

st.title("🚨 Detector de Anomalías: Lógica + Big-O + NumPy")
st.caption(
    "Módulo 4 — Matemáticas Discretas y Complejidad. "
    "La misma decisión lógica, evaluada de forma ingenua vs. vectorizada."
)

tab1, tab2, tab3 = st.tabs(
    ["🔎 Simulación de alarma", "📈 Notación Big-O", "⚡ Benchmark en vivo"]
)


# ---------------------------------------------------------------------------
# Utilidades compartidas
# ---------------------------------------------------------------------------
def generar_datos(n, seed=42):
    """
    Genera n lecturas sintéticas. Ahora incluye una tercera variable:
    el día de la semana (0 = lunes ... 6 = domingo), del que derivamos
    la proposición booleana `es_fin_de_semana`.
    """
    rng = np.random.default_rng(seed)
    temperaturas = rng.uniform(15, 40, n)
    humedades = rng.uniform(20, 80, n)
    dias = rng.integers(0, 7, n)
    es_fin_de_semana = dias >= 5  # sábado (5) y domingo (6) -> array de bool
    return temperaturas, humedades, es_fin_de_semana


def alarma_logica_loop(temperaturas, humedades, es_fin_de_semana,
                       temp_umbral, hum_umbral):
    """
    Versión ingenua: un elemento a la vez, con los operadores lógicos
    de Python puro (`and`, `not`).

        alarma = (temp > umbral) AND (hum < umbral) AND NOT(es_fin_de_semana)
    """
    resultados = []
    for temp, hum, finde in zip(temperaturas, humedades, es_fin_de_semana):
        resultados.append(temp > temp_umbral and hum < hum_umbral and not finde)
    return np.array(resultados)


def alarma_logica_vectorizada(temperaturas, humedades, es_fin_de_semana,
                              temp_umbral, hum_umbral):
    """
    Versión vectorizada: la misma expresión lógica sobre arrays completos.

    OJO con los operadores: en NumPy se usa `&` para AND y `~` para NOT
    (no `and` / `not`, que fallan sobre arrays). Además `~` solo significa
    negación lógica si el array es de tipo bool; sobre enteros haría
    complemento a nivel de bits.
    """
    return (temperaturas > temp_umbral) & (humedades < hum_umbral) & ~es_fin_de_semana


# ---------------------------------------------------------------------------
# Tab 1: Simulación de alarma
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Alarma por regla lógica")
    st.write(
        "La alarma dispara con una condición fija que tú defines, combinando tres "
        "proposiciones: *temperatura > umbral* **Y** *humedad < umbral* "
        "**Y NO** *es fin de semana*."
    )
    st.latex(r"A = (T > u_T) \;\wedge\; (H < u_H) \;\wedge\; \neg F")

    col_cfg, col_data = st.columns([1, 2])

    with col_cfg:
        n = st.slider("Número de lecturas (n)", 50, 5000, 500, step=50)
        temp_umbral = st.slider("Umbral temperatura (°C) — mayor que", 15, 40, 30)
        hum_umbral = st.slider("Umbral humedad (%) — menor que", 20, 80, 40)

    temps, hums, finde = generar_datos(n)

    # Separamos las dos primeras proposiciones de la tercera para poder
    # visualizar cuántas alarmas quedaron suprimidas por el NOT.
    cumple_sensores = (temps > temp_umbral) & (hums < hum_umbral)
    alarmas = cumple_sensores & ~finde
    suprimidas = cumple_sensores & finde

    with col_cfg:
        st.metric("Alarmas detectadas", f"{alarmas.sum()} / {n}")
        st.metric(
            "Suprimidas por fin de semana", f"{suprimidas.sum()}",
            help="Lecturas que cumplían temperatura y humedad, "
                 "pero que el NOT descartó por caer en sábado o domingo.",
        )

    with col_data:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.scatter(
            temps[~cumple_sensores], hums[~cumple_sensores],
            c="steelblue", alpha=0.5, label="Normal", s=15,
        )
        ax.scatter(
            temps[suprimidas], hums[suprimidas],
            c="darkorange", alpha=0.8, marker="x",
            label="Suprimida por fin de semana", s=35,
        )
        ax.scatter(
            temps[alarmas], hums[alarmas],
            c="crimson", alpha=0.8, label="Alarma / anomalía", s=25,
        )
        ax.axvline(temp_umbral, color="gray", ls="--", lw=1, alpha=0.6)
        ax.axhline(hum_umbral, color="gray", ls="--", lw=1, alpha=0.6)
        ax.set_xlabel("Temperatura (°C)")
        ax.set_ylabel("Humedad (%)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        st.pyplot(fig)

    st.caption(
        "Las cruces naranjas caen dentro del rectángulo de riesgo, pero la tercera "
        "proposición las desactiva. Es el efecto visual del operador NOT."
    )

    with st.expander("Ver datos y lógica aplicada"):
        df = pd.DataFrame({
            "temperatura": temps.round(2),
            "humedad": hums.round(2),
            "es_fin_de_semana": finde,
            "alarma": alarmas,
        })
        st.dataframe(df, use_container_width=True, height=250)


# ---------------------------------------------------------------------------
# Tab 2: Notación Big-O
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("¿Por qué importa la complejidad?")
    st.write(
        "El detector de arriba recorre `n` lecturas una vez: es un algoritmo **O(n)**. "
        "Aquí puedes ver qué tan distinto crece el número de operaciones frente a otras "
        "complejidades comunes, a medida que aumentan los datos."
    )

    n_max = st.slider("Tamaño máximo de n para la gráfica", 10, 200, 50)
    n_valores = np.arange(1, n_max + 1)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(n_valores, np.ones_like(n_valores), label="O(1) — constante")
    ax2.plot(n_valores, 2 * n_valores, ls=":", label="2n — versión de 2 condiciones")
    ax2.plot(n_valores, 3 * n_valores, ls="--", label="3n — versión de 3 condiciones")
    ax2.plot(n_valores, n_valores, label="O(n) — lineal (nuestro detector)")
    ax2.plot(n_valores, n_valores * np.log2(np.maximum(n_valores, 2)), label="O(n log n)")
    ax2.plot(n_valores, n_valores ** 2, label="O(n²) — cuadrática")
    ax2.set_xlabel("Tamaño de los datos (n)")
    ax2.set_ylabel("Operaciones (teórico)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    st.pyplot(fig2)

    st.info(
        "**¿Cambia el Big-O al agregar una tercera condición? No.** Pasamos de 2n a 3n "
        "comparaciones, pero 3n sigue siendo O(n): la notación descarta los factores "
        "constantes y solo describe *cómo escala* el costo cuando n crece. "
        "Las tres rectas punteadas de arriba tienen pendientes distintas, pero son "
        "todas rectas — ninguna se curva como O(n²). "
        "Agregar una cuarta, quinta o k-ésima condición seguiría siendo O(k·n) = O(n) "
        "mientras k sea fijo y no dependa de n."
    )


# ---------------------------------------------------------------------------
# Tab 3: Benchmark en vivo
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Loop vs. NumPy: misma lógica, distinta velocidad real")
    st.write(
        "Ejecuta la misma condición lógica de tres proposiciones sobre datos "
        "sintéticos, una vez con un loop de Python puro y otra vez vectorizada "
        "con NumPy, y compara el tiempo real."
    )

    n_bench = st.select_slider(
        "Tamaño de datos para el benchmark",
        options=[1_000, 10_000, 100_000, 500_000, 1_000_000],
        value=1_000_000,
    )
    temp_umbral_b = st.slider("Umbral temperatura (°C)", 15, 40, 30, key="temp_bench")
    hum_umbral_b = st.slider("Umbral humedad (%)", 20, 80, 40, key="hum_bench")

    if st.button("▶️ Ejecutar benchmark", type="primary"):
        temps_b, hums_b, finde_b = generar_datos(n_bench)

        # perf_counter tiene mucha más resolución que time.time(), y repetimos
        # varias veces porque la versión vectorizada puede ser demasiado rápida
        # para medirse de forma confiable en una sola corrida.
        repeticiones_loop = 1
        repeticiones_vec = 20

        inicio = time.perf_counter()
        for _ in range(repeticiones_loop):
            res_loop = alarma_logica_loop(
                temps_b, hums_b, finde_b, temp_umbral_b, hum_umbral_b
            )
        t_loop = (time.perf_counter() - inicio) / repeticiones_loop

        inicio = time.perf_counter()
        for _ in range(repeticiones_vec):
            res_vec = alarma_logica_vectorizada(
                temps_b, hums_b, finde_b, temp_umbral_b, hum_umbral_b
            )
        t_vec = (time.perf_counter() - inicio) / repeticiones_vec

        col1, col2, col3 = st.columns(3)
        col1.metric("Tiempo con loop", f"{t_loop*1000:.3f} ms")
        col2.metric("Tiempo con NumPy", f"{t_vec*1000:.3f} ms")

        if t_vec > 0:
            speedup = t_loop / t_vec
            col3.metric("NumPy es más rápido por", f"{speedup:,.0f}x")
        else:
            col3.metric("NumPy es más rápido por", "demasiado rápido para medir")

        # Comprobación de equivalencia lógica: las dos versiones deben
        # producir exactamente el mismo vector de booleanos.
        if np.array_equal(res_loop, res_vec):
            st.success(
                f"✅ Ambas versiones coinciden en las {int(res_vec.sum()):,} alarmas "
                f"detectadas sobre {n_bench:,} lecturas. Misma lógica, distinto costo."
            )
        else:
            st.error("⚠️ Las dos versiones no coinciden: revisa los operadores lógicos.")

        st.caption(
            f"Tiempo con loop promediado sobre {repeticiones_loop} corrida(s); "
            f"tiempo con NumPy promediado sobre {repeticiones_vec} corridas, "
            "para reducir el ruido de la medición."
        )

        fig3, ax3 = plt.subplots(figsize=(5, 3.5))
        ax3.bar(["Loop (Python)", "NumPy (vectorizado)"],
                [t_loop * 1000, t_vec * 1000],
                color=["indianred", "seagreen"])
        ax3.set_ylabel("Tiempo (milisegundos)")
        ax3.grid(alpha=0.3, axis="y")
        st.pyplot(fig3)
    else:
        st.caption("Ajusta los parámetros y presiona **Ejecutar benchmark** para ver el resultado.")
