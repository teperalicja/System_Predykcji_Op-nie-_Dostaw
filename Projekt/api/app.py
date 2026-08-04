from fastapi import FastAPI
from pydantic import BaseModel
from datetime import date
import pandas as pd
import joblib

app = FastAPI(title="Predykcja opóźnienia dostawy")

# Wczytanie modelu i artefaktów przy starcie serwera
model = joblib.load("../models/model_random_forest.pkl")
encoder = joblib.load("../models/target_encoder.pkl")
kolumny_modelu = joblib.load("../models/kolumny_modelu.pkl")

cechy_onehot = ['Tryb_wysylki', 'Rynek', 'Typ_platnosci', 'Nazwa_dzialu']
cechy_target_encoding = [
    'Miasto_klienta', 'Kraj_klienta', 'Stan_klienta',
    'Miasto_zamowienia', 'Kraj_zamowienia',
    'Nazwa_kategorii', 'Nazwa_produktu'
]

class Zamowienie(BaseModel):
    Typ_platnosci: str
    Planowane_dni_wysylki: int
    Tryb_wysylki: str
    Rynek: str
    Nazwa_kategorii: str
    Nazwa_dzialu: str
    Miasto_klienta: str
    Kraj_klienta: str
    Stan_klienta: str
    Miasto_zamowienia: str
    Kraj_zamowienia: str
    Nazwa_produktu: str
    Cena_produktu: float
    Ilosc_pozycji: int
    Rabat_pozycji: float
    Stopa_rabatu_pozycji: float
    Data_zamowienia: date


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/model/info")
def model_info():
    return {
        "model": "RandomForestClassifier",
        "liczba_cech": len(kolumny_modelu)
    }


@app.post("/predict")
def predict(zamowienie: Zamowienie):
    dane = zamowienie.dict()
    df = pd.DataFrame([dane])

    df['Data_zamowienia'] = pd.to_datetime(df['Data_zamowienia'])
    df['Dzien_tygodnia_zamowienia'] = df['Data_zamowienia'].dt.dayofweek
    df['Miesiac_zamowienia'] = df['Data_zamowienia'].dt.month
    df = df.drop(columns=['Data_zamowienia'])

    df_encoded = pd.get_dummies(df, columns=cechy_onehot)
    df_encoded = df_encoded.reindex(columns=kolumny_modelu, fill_value=0)

    df_encoded[cechy_target_encoding] = encoder.transform(df[cechy_target_encoding])

    predykcja = model.predict(df_encoded)[0]
    prawdopodobienstwo = model.predict_proba(df_encoded)[0][1]

    return {
        "opoznienie_przewidziane": bool(predykcja),
        "prawdopodobienstwo_opoznienia": round(float(prawdopodobienstwo), 4),
        "komunikat": "Przesyłka prawdopodobnie się opóźni" if predykcja == 1 else "Przesyłka prawdopodobnie dotrze na czas"
    }