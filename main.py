from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib
import nltk
import spacy
import re

app = FastAPI()

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"], 
)

model = load_model('mi_modelo.h5')
vectorizer_tfidf = joblib.load('vectorizer_tfidf.joblib')
train = pd.read_excel('ODScat_345.xlsx')

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('spanish'))
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

@app.post("/predict")
def predict(text_input: TextInput):
    textos_preprocesados = [preprocess_text(texto) for texto in text_input.textos]
    X_tfidf = vectorizer_tfidf.transform(textos_preprocesados)
    predictions = model.predict(X_tfidf)
    predicted_classes = np.argmax(predictions, axis=1) + 3
    probabilities = predictions.max(axis=1).tolist()
    resultados = [{"clase_predicha": int(clase), "probabilidad": float(prob)} for clase, prob in zip(predicted_classes, probabilities)]
    return resultados

@app.post("/retrain")
def retrain(retrain_input: RetrainInput):
    nuevos_datos = pd.DataFrame({"Textos_espanol": retrain_input.textos, "sdg": retrain_input.sdg})
    nuevos_datos['Textos_espanol'] = nuevos_datos['Textos_espanol'].apply(preprocess_text)
    datos_completos = pd.concat([train, nuevos_datos], ignore_index=True)
    X = vectorizer_tfidf.transform(datos_completos['Textos_espanol'])
    y = datos_completos['sdg'] - 3
    y_categorical = to_categorical(y, num_classes=3)
    X_train, X_val, y_train, y_val = train_test_split(X, y_categorical, test_size=0.2, random_state=42)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=32)
    model.save('mi_modelo.h5')
    y_pred = np.argmax(model.predict(X_val), axis=1)
    y_true = np.argmax(y_val, axis=1)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
