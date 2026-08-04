# 📦 System Predykcji Opóźnień Dostaw

### Alicja Teper
### Kurs: Data Science + AI

Projekt Data Science przewidujący ryzyko opóźnienia dostawy **w momencie tworzenia zamówienia** — na podstawie danych logistycznych, geograficznych i produktowych, bez wykorzystania informacji znanych dopiero po realizacji przesyłki.

> 🖥️ **Aplikacja działa lokalnie** 

## 📝 Opis

Firmy logistyczne i e-commerce tracą zaufanie klientów, gdy przesyłki docierają później niż obiecano. Ten projekt odpowiada na pytanie:

> **Czy nowo składane zamówienie prawdopodobnie dotrze z opóźnieniem — jeszcze zanim zostanie wysłane?**

W przeciwieństwie do prostego liczenia opóźnień na gotowych, historycznych danych, aplikacja pozwala **wprowadzić dane nowego zamówienia** (produkt, trasa dostawy, tryb wysyłki, klient) i natychmiast otrzymać przewidywane ryzyko opóźnienia wraz z prawdopodobieństwem.

## 🎯 Problem biznesowy

- **Pytanie badawcze:** Czy na podstawie danych dostępnych w momencie składania zamówienia można przewidzieć ryzyko jego opóźnienia?
- **Metryka sukcesu:** Model istotnie przewyższający baseline (54.8% accuracy) oraz zbalansowane wykrywanie obu klas (opóźnione / na czas)
- **Wynik:** Random Forest osiągający **77.99% accuracy** (pojedynczy split) / **77.46% ± 0.21%** (5-krotna walidacja krzyżowa) i **AUC = 0.865**

## 📊 Dane

- **Źródło:** [DataCo Smart Supply Chain Dataset]
- **Wielkość:** ~180 500 zamówień, 53 kolumny (oryginalnie)
- **Target:** `Ryzyko_opoznienia` (binarny: 0 = na czas, 1 = opóźnione), balans klas 54.8% / 45.2%

Kluczowa decyzja projektowa: z modelu **celowo wykluczono** kolumny nieznane w momencie składania zamówienia (np. rzeczywiste dni wysyłki, status dostawy, data wysyłki, zysk na zamówienie). Model wykorzystuje **wyłącznie** cechy dostępne *przed* realizacją przesyłki — dokładnie te, które użytkownik podaje w formularzu aplikacji.

## 🔬 Metodologia

| Etap | Opis |
| **EDA** | Analiza rozkładów, korelacji, balansu klas |
| **Feature engineering** | Cechy czasowe (dzień tygodnia, miesiąc), one-hot encoding (tryb wysyłki, rynek), **target encoding** dla cech wysokiej kardynalności (miasto, kraj, produkt — do 3597 unikalnych wartości) |
| **Modelowanie** | Baseline (`DummyClassifier`) → porównanie 4 algorytmów → 5-krotna walidacja krzyżowa finalistów |
| **Walidacja statystyczna** | Sparowany test t-Studenta na wynikach cross-validation — potwierdzenie, że przewaga wybranego modelu nie jest przypadkowa |
| **Interpretowalność** | SHAP (SHapley Additive exPlanations) — wyjaśnienie predykcji na poziomie całego modelu i pojedynczych zamówień |
| **Ewaluacja** | Confusion matrix, krzywa ROC/AUC, krzywa Precision-Recall, feature importance |

### Porównanie modeli (pojedynczy train/test split)

| Model | Accuracy | Czas treningu |
| Baseline (Dummy) | 0.5483 | – |
| Logistic Regression | 0.7036 | ~2 s |
| **Random Forest** ⭐ | **0.7799** | ~70–110 s |
| Gradient Boosting | 0.7049 | ~120–240 s |

**Wybrany model:** Random Forest — najlepsza dokładność przy rozsądnym czasie treningu. Gradient Boosting z domyślnymi parametrami nie uzasadnił swojego znacznie dłuższego czasu treningu.

### Rzetelność wyniku: cross-validation i istotność statystyczna

Pojedynczy podział danych (train/test) może dać wynik zależny od przypadkowego trafienia. Dlatego dla dwóch finalistów przeprowadzono **5-krotną walidację krzyżową** (`StratifiedKFold`):

| Model | Średnia accuracy (CV) | Odchylenie std. |
| **Random Forest** | **0.7746** | ± 0.0021 |
| Logistic Regression | 0.7153 | ± ~0.003 |

Bardzo małe odchylenie standardowe (±0.21 p.p.) potwierdza, że wynik Random Forest jest **stabilny** niezależnie od podziału danych.

Różnicę między modelami zweryfikowano **sparowanym testem t-Studenta**: t = 53.59, **p = 0.000001** (p ≪ 0.05) — przewaga Random Forest nad Logistic Regression jest statystycznie istotna, a nie efektem przypadku.

### Interpretowalność: SHAP

Analiza SHAP na modelu Random Forest wyjaśnia, jak poszczególne cechy wpływają na predykcję:

- **Na poziomie całego modelu** (`wykresy/11_shap_summary.png`) — tryb wysyłki i planowany czas dostawy mają największy wpływ; zaskakująco, przesyłki **First Class** wykazują *wyższe* ryzyko opóźnienia niż **Standard Class**, co może sugerować przeciążenie szybszych kanałów logistycznych.
- **Na poziomie pojedynczego zamówienia** (`wykresy/12_shap_waterfall_przyklad.png`) — wykres typu waterfall pokazuje krok po kroku, które cechy konkretnego zamówienia podniosły lub obniżyły jego ryzyko względem wartości bazowej (54.8%).

### Najważniejsze cechy (feature importance)

Geografia dostawy (miasto/kraj zamówienia i klienta) odpowiada łącznie za **~34%** ważności cech — sugeruje to, że opóźnienia są w większym stopniu problemem **logistycznym**, niż związanym z samym produktem. Analiza SHAP potwierdza ten wniosek i dodatkowo uwypukla rolę trybu wysyłki.

### Tracking eksperymentów: MLflow

Wszystkie 4 modele (Baseline, Logistic Regression, Random Forest, Gradient Boosting) zostały zalogowane w **MLflow** — wraz z pełnymi parametrami, metrykami (accuracy, czas treningu) oraz wersją każdego wytrenowanego modelu. Zapewnia to pełną powtarzalność i możliwość audytu procesu modelowania.

Podgląd lokalnie: `mlflow ui` (uruchomione z folderu `Projekty/`), następnie `http://localhost:5000` → zakładka **Model training**.

## 🏗️ Architektura aplikacji

Użytkownik → Formularz Streamlit → REST API (FastAPI) → Model (Random Forest + Target Encoder) → Predykcja

- **Backend:** FastAPI z endpointami `/predict`, `/health`, `/model/info`
- **Frontend:** Streamlit — formularz "stwórz zamówienie" z natychmiastową predykcją i wizualizacją prawdopodobieństwa

## 📁 Struktura projektu

├── api/
│   └── app.py                 # Backend FastAPI
├── models/
│   ├── model_random_forest.pkl
│   ├── target_encoder.pkl
│   ├── kolumny_modelu.pkl
│   └── metadata.json
├── notebooks/
│   └── System_Predykcji_Opóźnień_Dostaw.ipynb
├── wykresy/                   # Wizualizacje EDA i ewaluacji
├── streamlit_app.py           # Frontend
├── requirements.txt
└── README.md


## 🚀 Uruchomienie

```bash
# Instalacja zależności
pip install -r requirements.txt

# Backend (terminal 1)
cd api
uvicorn app:app --reload

# Frontend (terminal 2)
streamlit run streamlit_app.py
```

Aplikacja dostępna pod `http://localhost:8501`, dokumentacja API pod `http://localhost:8000/docs`.

## 🌱 Green IT

Świadomy wybór modelu z uwzględnieniem efektywności: Gradient Boosting wymagał o **26% dłuższego** czasu treningu niż Random Forest, oferując przy tym **gorszy** wynik accuracy — Random Forest okazał się wyborem efektywnym zarówno dokładnościowo, jak i obliczeniowo.

## 🛠️ Technologie

Python · pandas · scikit-learn · category_encoders · FastAPI · Streamlit · matplotlib · seaborn

