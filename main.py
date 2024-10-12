from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np
import pandas as pd
import joblib
import nltk
import spacy
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

app = FastAPI()

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"], 
)

# Load the pipeline that contains both the vectorizer and the model
pipeline = joblib.load('pipeline.pkl')
train = pd.read_excel('ODScat_345.xlsx')
# Dividir el conjunto de datos


# Load the training data


# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('spanish'))

# Load spaCy model
nlp = spacy.load('es_core_news_sm')

def corregir_caracteres(texto):
    correcciones = {'Ãº': 'ú', 'ãº': 'ú', 'Ã³': 'ó', 'Ã©': 'é', 'Ã±': 'ñ', 'Ã¡': 'á', 'Ã­': 'í'}
    for incorrecto, correcto in correcciones.items():
        texto = texto.replace(incorrecto, correcto)
    return texto

def eliminar_consecutivas(texto):
    palabras = texto.split()
    palabras_filtradas = [palabra for i, palabra in enumerate(palabras) if i == 0 or palabra != palabras[i - 1]]
    return ' '.join(palabras_filtradas)

def ajustar_texto(texto):
    texto = texto.replace('%', ' porcentaje')
    return re.sub(r'(\d+)[.,](\d+)', lambda x: str(round(float(x.group().replace(',', '.')))), texto)

def procesar_texto(texto):
    tokens = nltk.word_tokenize(texto, language='spanish')
    return [token for token in tokens if token.lower() not in stop_words]

def lematizar_tokens(tokens):
    doc = spacy.tokens.Doc(nlp.vocab, words=tokens)
    for name, proc in nlp.pipeline:
        doc = proc(doc)
    return [token.lemma_ for token in doc]

def preprocess_text(texto):
    texto = corregir_caracteres(texto)
    texto = eliminar_consecutivas(texto)
    texto = ajustar_texto(texto)
    tokens = procesar_texto(texto)
    lemas = lematizar_tokens(tokens)
    return ' '.join(lemas)

class TextInput(BaseModel):
    textos: List[str]

class RetrainInput(BaseModel):
    textos: List[str]
    sdg: List[int]

X_train, X_test, y_train, y_test = train_test_split(train['Textos_espanol'], train['sdg'], test_size=0.3, random_state=42)

# Entrenar el modelo usando el pipeline
pipeline.fit(X_train, y_train)
y_test_pred_rf = pipeline.predict(X_test)

print("\nTesting Classification Report:\n", classification_report(y_test, y_test_pred_rf))

@app.post("/predict")
def predict(text_input: TextInput):
    textos_preprocesados = [preprocess_text(texto) for texto in text_input.textos]

    predicted_probabilities = pipeline.predict_proba(textos_preprocesados)
    predicted_classes = pipeline.predict(textos_preprocesados)
    probabilities = predicted_probabilities.max(axis=1).tolist()
    resultados = [{"clase_predicha": int(clase), "probabilidad": float(prob)} for clase, prob in zip(predicted_classes, probabilities)]
    return resultados

@app.post("/retrain")
def retrain(retrain_input: RetrainInput):
    # Preprocess new data
    nuevos_datos = pd.DataFrame({"Textos_espanol": retrain_input.textos, "sdg": retrain_input.sdg})
    nuevos_datos['Textos_espanol'] = nuevos_datos['Textos_espanol'].apply(preprocess_text)

    # Combine with existing training data
    datos_completos = pd.concat([train, nuevos_datos], ignore_index=True)
    X = datos_completos['Textos_espanol']
    y = datos_completos['sdg']

    # Split data for validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Retrain the pipeline on the training data
    pipeline.fit(X_train, y_train)

    # Predict on validation data
    y_pred = pipeline.predict(X_val)

    # Calculate performance metrics
    precision = precision_score(y_val, y_pred, average='weighted')
    recall = recall_score(y_val, y_pred, average='weighted')
    f1 = f1_score(y_val, y_pred, average='weighted')

    # Retrain on the full dataset
    pipeline.fit(X, y)

    # Save the updated pipeline
    joblib.dump(pipeline, 'pipeline.pkl')

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }