import streamlit as st
import requests

st.set_page_config(page_title="Predykcja opóźnienia dostawy", page_icon="📦")

st.title("📦 Stwórz zamówienie")
st.write("Wypełnij dane zamówienia, aby sprawdzić przewidywane ryzyko opóźnienia dostawy.")

API_URL = "http://localhost:8000/predict"

# Listy opcji na podstawie danych treningowych (uzupełnij wg swojego datasetu)
TRYBY_WYSYLKI = ["Standard Class", "First Class", "Second Class", "Same Day"]
RYNKI = ["Pacific Asia", "Europe", "USCA", "LATAM", "Africa"]
TYPY_PLATNOSCI = ["DEBIT", "TRANSFER", "CASH", "PAYMENT"]

with st.form("formularz_zamowienia"):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Dane produktu")
        nazwa_produktu = st.text_input("Nazwa produktu", "Smart watch")
        nazwa_kategorii = st.text_input("Kategoria produktu", "Sporting Goods")
        nazwa_dzialu = st.text_input("Dział", "Fan Shop")
        cena_produktu = st.number_input("Cena produktu", min_value=0.0, value=100.0)
        ilosc_pozycji = st.number_input("Ilość", min_value=1, value=1, step=1)
        rabat_pozycji = st.number_input("Rabat (kwota)", min_value=0.0, value=0.0)
        stopa_rabatu = st.number_input("Stopa rabatu (0-1)", min_value=0.0, max_value=1.0, value=0.0)

    with col2:
        st.subheader("Dane dostawy")
        typ_platnosci = st.selectbox("Typ płatności", TYPY_PLATNOSCI)
        tryb_wysylki = st.selectbox("Tryb wysyłki", TRYBY_WYSYLKI)
        planowane_dni = st.number_input("Planowane dni wysyłki", min_value=1, value=4, step=1)
        rynek = st.selectbox("Rynek", RYNKI)
        data_zamowienia = st.date_input("Data zamówienia")

        st.subheader("Dane klienta i lokalizacji")
        miasto_klienta = st.text_input("Miasto klienta", "Caguas")
        kraj_klienta = st.text_input("Kraj klienta", "Puerto Rico")
        stan_klienta = st.text_input("Stan klienta", "PR")
        miasto_zamowienia = st.text_input("Miasto dostawy", "San Jose")
        kraj_zamowienia = st.text_input("Kraj dostawy", "Estados Unidos")

    submitted = st.form_submit_button("🔮 Sprawdź ryzyko opóźnienia")

if submitted:
    payload = {
        "Typ_platnosci": typ_platnosci,
        "Planowane_dni_wysylki": int(planowane_dni),
        "Tryb_wysylki": tryb_wysylki,
        "Rynek": rynek,
        "Nazwa_kategorii": nazwa_kategorii,
        "Nazwa_dzialu": nazwa_dzialu,
        "Miasto_klienta": miasto_klienta,
        "Kraj_klienta": kraj_klienta,
        "Stan_klienta": stan_klienta,
        "Miasto_zamowienia": miasto_zamowienia,
        "Kraj_zamowienia": kraj_zamowienia,
        "Nazwa_produktu": nazwa_produktu,
        "Cena_produktu": float(cena_produktu),
        "Ilosc_pozycji": int(ilosc_pozycji),
        "Rabat_pozycji": float(rabat_pozycji),
        "Stopa_rabatu_pozycji": float(stopa_rabatu),
        "Data_zamowienia": str(data_zamowienia)
    }

    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        wynik = response.json()

        st.divider()
        if wynik["opoznienie_przewidziane"]:
            st.error(f"⚠️ {wynik['komunikat']}")
        else:
            st.success(f"✅ {wynik['komunikat']}")

        st.metric("Prawdopodobieństwo opóźnienia",
                   f"{wynik['prawdopodobienstwo_opoznienia'] * 100:.1f}%")
        st.progress(wynik['prawdopodobienstwo_opoznienia'])

    except requests.exceptions.ConnectionError:
        st.error("❌ Nie udało się połączyć z API. Upewnij się, że backend FastAPI działa (uvicorn app:app --reload w folderze api/).")
    except Exception as e:
        st.error(f"❌ Błąd: {e}")