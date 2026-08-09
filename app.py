import os, json, joblib
import pandas as pd
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title='Riesgo actuarial', layout='centered')
st.title('Predicción de riesgo actuarial-Oscar Noe Espino-PTI-0620')

@st.cache_resource
def cargar_modelo():
    pkl = 'kmeans_riesgo_actuarial.pkl' if os.path.exists('kmeans_riesgo_actuarial.pkl') else 'kmeans_riesgo_actuarial(2).pkl'
    meta = 'model_metadata.json' if os.path.exists('model_metadata.json') else 'model_metadata(2).json'
    
    modelo = joblib.load(pkl)

    with open(meta, encoding='utf-8') as f:
        metadata = json.load(f)

    return modelo, metadata

@st.cache_data
def cargar_base():
    csv = 'insurance.csv' if os.path.exists('insurance.csv') else 'insurance(2).csv'
    return pd.read_csv(csv)

modelo, metadata = cargar_modelo()
df = cargar_base()


def obtener_mapa_riesgo(metadata: dict) -> dict:
    """Busca el mapa de riesgo en varias ubicaciones comunes del archivo de metadatos.

    Soporta mapas como diccionarios (con claves numéricas o strings) o listas
    (se convierten a mapeos por índice). Devuelve un mapa por defecto si no
    se encuentra ninguno.
    """
    claves_posibles = [
        'mapa_riesgo',
        'mapa_clases',
        'clases',
        'class_mapping',
        'etiquetas',
        'labels',
    ]

    # Buscar en el nivel superior y dentro de secciones habituales
    for clave in claves_posibles:
        valor = metadata.get(clave)

        if valor is None:
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
                    clave_convertida = int(k)
                except (TypeError, ValueError):
                    clave_convertida = str(k)
                mapa[clave_convertida] = str(v)
            return mapa

        if isinstance(valor, list):
            return {i: str(etiqueta) for i, etiqueta in enumerate(valor)}

    # Predeterminado si no se encuentra
    return {0: 'Bajo', 1: 'Medio', 2: 'Alto'}


mapa = obtener_mapa_riesgo(metadata)

st.caption(metadata.get('nombre_modelo', metadata.get('proyecto', '')))

with st.form('datos'):

    col1, col2 = st.columns(2)

    age = col1.number_input('Edad', 18, 100, 35)
    sex = col2.selectbox('Sexo', sorted(df['sex'].unique()))

    bmi = col1.number_input('BMI', 10.0, 60.0, 28.0)
    children = col2.number_input('Hijos', 0, 10, 1)

    smoker = col1.selectbox('Fumador', sorted(df['smoker'].unique()))
    region = col2.selectbox('Región', sorted(df['region'].unique()))

    charges = st.number_input(
        'Cargos médicos estimados',
        0.0,
        100000.0,
        12000.0
    )

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

    cluster = int(modelo.predict(cliente)[0])

    riesgo = mapa.get(cluster, 'No definido')

    st.subheader(f'Riesgo actuarial: {riesgo}')
    st.write(f'Cluster asignado: {cluster}')

    api_key = st.secrets.get(
        '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',
        os.getenv('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', '')
    )

    if api_key:

        prompt = f'''
        Actúa como analista actuarial.

        Explica brevemente el resultado y brinda
        3 recomendaciones prudentes.

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

            client = OpenAI(
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1"
            )

            respuesta = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": "Eres un asesor actuarial profesional."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            texto = respuesta.choices[0].message.content

            st.info(texto)

        except Exception as e:

            st.warning(f'Error con Groq: {e}')

    else:

        st.warning(
            'GROQ_API_KEY="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"'
        )

st.divider()

st.write('Vista rápida de la base principal')

st.dataframe(
    df.head(20),
    use_container_width=True
)
