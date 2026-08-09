import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================
st.set_page_config(
    page_title="Predicción de riesgo actuarial",
    page_icon="📊",
    layout="centered",
)

st.title("Predicción de riesgo actuarial - NahunFlores PTI-0620")
st.write(
    "Ingrese los datos solicitados para estimar el nivel de riesgo actuarial."
)


# ============================================================
# RUTAS DE ARCHIVOS
# Cambie estos nombres únicamente si sus archivos se llaman distinto.
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "modelo.joblib"
METADATA_PATH = BASE_DIR / "metadata.json"


# ============================================================
# FUNCIONES DE CARGA
# ============================================================
@st.cache_resource
def cargar_modelo(ruta: Path):
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo: {ruta.name}. "
            "Verifique que esté en la misma carpeta que app.py."
        )
    return joblib.load(ruta)


@st.cache_data
def cargar_metadata(ruta: Path) -> dict:
    if not ruta.exists():
        return {}

    with ruta.open("r", encoding="utf-8") as archivo:
        datos = json.load(archivo)

    if not isinstance(datos, dict):
        raise ValueError("metadata.json debe contener un objeto JSON.")

    return datos


def obtener_variables(modelo, metadata: dict) -> list[str]:
    """
    Busca los nombres de variables en metadata.json.
    Si no aparecen, intenta obtenerlas del modelo de scikit-learn.
    """
    claves_posibles = [
        "variables",
        "features",
        "feature_names",
        "columnas",
        "campos",
    ]

    for clave in claves_posibles:
        valor = metadata.get(clave)
        if isinstance(valor, list) and valor:
            return [str(x) for x in valor]

    if hasattr(modelo, "feature_names_in_"):
        return [str(x) for x in modelo.feature_names_in_]

    return []


def obtener_mapa_riesgo(metadata: dict) -> dict:
    """
    Evita el KeyError cuando metadata.json no contiene 'mapa_riesgo'.
    Acepta varios nombres habituales.
    """
    claves_posibles = [
        "mapa_riesgo",
        "clases",
        "class_mapping",
        "mapa_clases",
        "etiquetas",
        "labels",
    ]

    for clave in claves_posibles:
        valor = metadata.get(clave)

        if isinstance(valor, dict):
            mapa = {}
            for k, v in valor.items():
                try:
                    clave_convertida = int(k)
                except (TypeError, ValueError):
                    clave_convertida = str(k)

                mapa[clave_convertida] = str(v)

            return mapa

        if isinstance(valor, list):
            return {i: str(etiqueta) for i, etiqueta in enumerate(valor)}

    # Valores predeterminados. Se usan solamente si metadata.json
    # no incluye ningún mapa de clases.
    return {
        0: "Bajo",
        1: "Medio",
        2: "Alto",
    }


def obtener_configuracion_variable(metadata: dict, variable: str) -> dict:
    """
    Permite configurar cada campo desde metadata.json.
    Ejemplo:
    "configuracion_variables": {
        "edad": {"tipo": "int", "min": 18, "max": 100, "default": 35}
    }
    """
    configuraciones = metadata.get(
        "configuracion_variables",
        metadata.get("feature_config", {}),
    )

    if isinstance(configuraciones, dict):
        config = configuraciones.get(variable, {})
        if isinstance(config, dict):
            return config

    return {}


def crear_campo(variable: str, metadata: dict):
    config = obtener_configuracion_variable(metadata, variable)

    etiqueta = str(config.get("etiqueta", variable.replace("_", " ").title()))
    tipo = str(config.get("tipo", "float")).lower()
    ayuda = config.get("ayuda")

    opciones = config.get("opciones")
    if isinstance(opciones, list) and opciones:
        return st.selectbox(etiqueta, opciones=opciones, help=ayuda)

    if tipo in {"int", "integer", "entero"}:
        minimo = int(config.get("min", 0))
        maximo = int(config.get("max", 100))
        valor = int(config.get("default", minimo))
        paso = int(config.get("step", 1))

        return st.number_input(
            etiqueta,
            min_value=minimo,
            max_value=maximo,
            value=valor,
            step=paso,
            help=ayuda,
        )

    if tipo in {"bool", "boolean", "booleano"}:
        return int(st.checkbox(etiqueta, help=ayuda))

    minimo = float(config.get("min", 0.0))
    maximo = float(config.get("max", 1_000_000.0))
    valor = float(config.get("default", minimo))
    paso = float(config.get("step", 1.0))

    return st.number_input(
        etiqueta,
        min_value=minimo,
        max_value=maximo,
        value=valor,
        step=paso,
        help=ayuda,
    )


def traducir_prediccion(prediccion, mapa: dict) -> str:
    if prediccion in mapa:
        return mapa[prediccion]

    texto = str(prediccion)

    if texto in mapa:
        return mapa[texto]

    try:
        entero = int(prediccion)
        if entero in mapa:
            return mapa[entero]
    except (TypeError, ValueError):
        pass

    return texto


# ============================================================
# CARGA DEL MODELO Y LOS METADATOS
# ============================================================
try:
    modelo = cargar_modelo(MODEL_PATH)
    metadata = cargar_metadata(METADATA_PATH)
except Exception as error:
    st.error(f"No fue posible iniciar la aplicación: {error}")
    st.stop()


variables = obtener_variables(modelo, metadata)
mapa_riesgo = obtener_mapa_riesgo(metadata)

if not variables:
    st.error(
        "No fue posible determinar las variables requeridas por el modelo. "
        "Agregue una lista llamada 'variables' en metadata.json o entrene el "
        "modelo con nombres de columnas."
    )
    st.stop()


# ============================================================
# FORMULARIO
# ============================================================
with st.form("formulario_prediccion"):
    st.subheader("Datos para la predicción")

    valores = {}
    for variable in variables:
        valores[variable] = crear_campo(variable, metadata)

    enviar = st.form_submit_button(
        "Calcular riesgo",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# PREDICCIÓN
# ============================================================
if enviar:
    try:
        entrada = pd.DataFrame([valores], columns=variables)
        prediccion = modelo.predict(entrada)[0]
        nivel = traducir_prediccion(prediccion, mapa_riesgo)

        st.success(f"Nivel de riesgo estimado: {nivel}")

        if hasattr(modelo, "predict_proba"):
            probabilidades = modelo.predict_proba(entrada)[0]
            clases = getattr(
                modelo,
                "classes_",
                list(range(len(probabilidades))),
            )

            tabla = pd.DataFrame(
                {
                    "Nivel": [
                        traducir_prediccion(clase, mapa_riesgo)
                        for clase in clases
                    ],
                    "Probabilidad": probabilidades,
                }
            )

            tabla["Probabilidad"] = tabla["Probabilidad"].map(
                lambda x: f"{x:.2%}"
            )

            st.subheader("Probabilidades")
            st.dataframe(
                tabla,
                hide_index=True,
                use_container_width=True,
            )

        with st.expander("Datos enviados al modelo"):
            st.dataframe(
                entrada,
                hide_index=True,
                use_container_width=True,
            )

    except Exception as error:
        st.error(f"No fue posible realizar la predicción: {error}")
        st.info(
            "Revise que las variables, el orden de las columnas y los tipos "
            "de datos coincidan con los utilizados durante el entrenamiento."
        )


# ============================================================
# INFORMACIÓN TÉCNICA
# ============================================================
with st.sidebar:
    st.header("Información")

    st.write(f"Modelo: `{MODEL_PATH.name}`")

    if METADATA_PATH.exists():
        st.write(f"Metadatos: `{METADATA_PATH.name}`")
    else:
        st.warning(
            "No se encontró metadata.json. La aplicación está usando "
            "configuraciones predeterminadas."
        )

    st.caption(
        "El resultado es una estimación del modelo y debe interpretarse "
        "junto con criterios técnicos y actuariales."
    )
