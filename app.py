import os
import json
import joblib
import pandas as pd
import streamlit as st

# Opcional: cliente OpenAI/Groq si se quiere usar
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

st.set_page_config(page_title='Riesgo actuarial', layout='centered')
st.title('Predicción de riesgo actuarial - Oscar Noe Espino - PTI-0620')

# Rutas
BASE_DIR = os.path.dirname(__file__)
MODEL_PKL = os.path.join(BASE_DIR, 'kmeans_riesgo_actuarial.pkl')
METADATA_JSON = os.path.join(BASE_DIR, 'model_metadata.json')
CSV_DB = os.path.join(BASE_DIR, 'insurance.csv')

@st.cache_resource
def cargar_modelo(ruta_pkl=MODEL_PKL, ruta_meta=METADATA_JSON):
    if not os.path.exists(ruta_pkl):
        raise FileNotFoundError(f"No se encontró el modelo: {ruta_pkl}")
    modelo = joblib.load(ruta_pkl)

    metadata = {}
    if os.path.exists(ruta_meta):
        with open(ruta_meta, encoding='utf-8') as f:
            metadata = json.load(f)

    return modelo, metadata

@st.cache_data
def cargar_base(ruta_csv=CSV_DB):
    if os.path.exists(ruta_csv):
        return pd.read_csv(ruta_csv)
    return pd.DataFrame()

modelo, metadata = cargar_modelo()
df = cargar_base()


def obtener_mapa_riesgo(metadata: dict) -> dict:
    """Busca el mapa de riesgo en varias ubicaciones comunes del archivo de metadatos.

    Si no lo encuentra, devuelve un mapa por defecto.
    """
    claves_posibles = [
        'mapa_riesgo', 'mapa_clases', 'clases', 'class_mapping', 'etiquetas', 'labels'
    ]

    for clave in claves_posibles:
        valor = metadata.get(clave)

        if valor is None:
            # buscar en secciones anidadas (ej. kmeans, svm)
            for padre in ('kmeans', 'svm'):
                padre_dict = metadata.get(padre)
                if isinstance(padre_dict, dict):
                    valor = padre_dict.get(clave)
                    if valor is not None:
                        break

        if isinstance(valor, dict):
            mapa = {}
            for k, v in valor.items():
                try:
                    k_conv = int(k)
                except (TypeError, ValueError):
                    k_conv = str(k)
                mapa[k_conv] = str(v)
            return mapa

        if isinstance(valor, list):
            return {i: str(etiqueta) for i, etiqueta in enumerate(valor)}

    # fallback
    return {0: 'Bajo', 1: 'Medio', 2: 'Alto'}

mapa = obtener_mapa_riesgo(metadata)

st.caption(metadata.get('proyecto', ''))

with st.form('datos'):
    col1, col2 = st.columns(2)

    age = col1.number_input('Edad', 18, 100, 35)
    sex = col2.selectbox('Sexo', sorted(df['sex'].unique()) if not df.empty else ['male', 'female'])

    bmi = col1.number_input('BMI', 10.0, 60.0, 28.0)
    children = col2.number_input('Hijos', 0, 10, 1)

    smoker = col1.selectbox('Fumador', sorted(df['smoker'].unique()) if not df.empty else ['yes', 'no'])
    region = col2.selectbox('Región', sorted(df['region'].unique()) if not df.empty else ['southwest','southeast','northwest','northeast'])

    charges = st.number_input('Cargos médicos estimados', 0.0, 100000.0, 12000.0)

    enviar = st.form_submit_button('Evaluar')

if enviar:
    cliente = pd.DataFrame([{
        'age': age,
        'sex': sex,
        'bmi': bmi,
        'children': children,
        'smoker': smoker,
        'region': region,
        'charges': charges
    }])

    try:
        cluster = int(modelo.predict(cliente)[0])
    except Exception as e:
        st.error(f'Error al predecir: {e}')
        cluster = None

    riesgo = mapa.get(cluster, 'No definido') if cluster is not None else 'No definido'

    st.subheader(f'Riesgo actuarial: {riesgo}')
    if cluster is not None:
        st.write(f'Cluster asignado: {cluster}')

    # Leer clave segura sin exponerla
    api_key = None

    # Preferir st.secrets (Streamlit Cloud) luego VARIABLE DE ENTORNO
    try:
        api_key = st.secrets.get('GROQ_API_KEY') if hasattr(st, 'secrets') else None
    except Exception:
        api_key = None

    if not api_key:
        api_key = os.environ.get('GROQ_API_KEY')

    if api_key and OpenAI is not None:
        prompt = f'''
Actúa como analista actuarial.

Explica brevemente el resultado y brinda 3 recomendaciones prudentes.

Datos:
edad={age}
sexo={sex}
bmi={bmi}
hijos={children}
fumador={smoker}
región={region}
cargos={charges}

Resultado:
cluster={cluster}
riesgo={riesgo}
'''
        try:
            client = OpenAI(api_key=api_key, base_url='https://api.groq.com/openai/v1')
            respuesta = client.chat.completions.create(
                model='llama-3.1-8b-instant',
                messages=[
                    {"role": "system", "content": "Eres un asesor actuarial profesional."},
                    {"role": "user", "content": prompt}
                ]
            )
            texto = respuesta.choices[0].message.content
            st.info(texto)
        except Exception as e:
            st.warning(f'Error al consultar el servicio de LLM: {e}')
    else:
        st.info('Nota: no se configuró GROQ_API_KEY. Si deseas respuestas automáticas, añade la clave en Secrets (Streamlit) o en la variable de entorno GROQ_API_KEY.')

st.divider()
st.write('Vista rápida de la base principal')
if not df.empty:
    st.dataframe(df.head(20), use_container_width=True)
else:
    st.info('No se encontró la base de datos local (insurance.csv).')
