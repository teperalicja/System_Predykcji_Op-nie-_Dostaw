import os
import sys

# Dynamiczne ustawienie ścieżki roboczej do folderu z tym plikiem
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
os.chdir(current_dir)

import streamlit as st
import pandas as pd
from model_ml import SupplyChainML
from agent_llm import SupplyChainLLM

# Konfiguracja strony
st.set_page_config(page_title="Supply Chain AI Monitor", layout="wide", initial_sidebar_state="expanded")

st.title("🛡️ System Monitorowania Ryzyka Opóźnień w Łańcuchu Dostaw")
st.caption("Projekt zaliczeniowy: Integracja klasycznego Machine Learning z LLM (Agentic Decision Framework)")

@st.cache_resource
def load_systems():
    df = pd.read_csv(r'data/DataCoSupplyChainDataset.csv', encoding='latin1')
    if 'Order Id' not in df.columns:
        df['Order Id'] = df.index
    
    # Trening klasyfikatora ML
    ml_system = SupplyChainML()
    ml_system.train(df)
    
    # Inicjalizacja Agenta LLM
    llm_system = SupplyChainLLM()
    return df, ml_system, llm_system

df, ml_system, llm_system = load_systems()

# --- PANEL BOCZNY ---
st.sidebar.header("🎛️ Panel Kontrolny")
order_ids = sorted(df['Order Id'].unique())
selected_order_id = st.sidebar.selectbox("Wybierz ID Zlecenia do testu:", order_ids[:500]) # Ograniczenie dla wygody

# Wyciągnięcie danych pojedynczego zlecenia
order_data = df[df['Order Id'] == selected_order_id].iloc[0]

st.sidebar.markdown("---")
st.sidebar.subheader("📋 Metadane Zamówienia")
st.sidebar.write(f"**ID Zamówienia:** `{selected_order_id}`")
st.sidebar.write(f"**Produkt:** {order_data['Product Name']}")
st.sidebar.write(f"**Cena produktu:** {order_data['Product Price']} USD")
st.sidebar.write(f"**Zaplanowane dni wysyłki:** {order_data['Days for shipment (scheduled)']}")

# --- GŁÓWNA SEKCJA ANALIZY ---
st.subheader("📊 Krok 1: Predykcja Ryzyka (Model ML - Random Forest)")

if st.button("Uruchom analizę potoku decyzyjnego"):
    # 1. Obliczenie ryzyka za pomocą modelu ML
    ryzyko = ml_system.predict_risk(order_data)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric(
            label="Prawdopodobieństwo opóźnienia", 
            value=f"{ryzyko:.1%}",
            delta="- Bezpiecznie" if ryzyko <= 0.50 else "+ WYSOKIE RYZYKO",
            delta_color="inverse"
        )
    with col2:
        # Tabela z cechami wejściowymi dla modelu (świetnie wygląda w dokumentacji projektu)
        features_df = pd.DataFrame({
            "Cecha (Feature)": [
                "Planowane dni na wysyłkę", 
                "Zysk z zamówienia", 
                "Wielkość sprzedaży per klient", 
                "Cena jednostkowa"
            ],
            "Wartość dla zlecenia": [
                f"{order_data['Days for shipment (scheduled)']} dni",
                f"{order_data['Benefit per order']:.2f} USD",
                f"{order_data['Sales per customer']:.2f} USD",
                f"{order_data['Product Price']:.2f} USD"
            ]
        })
        st.table(features_df)

    st.markdown("---")
    st.subheader("🤖 Krok 2: Decyzja Agenta LLM (System Rekomendacji Decyzyjnych)")

    if ryzyko > 0.50:
        st.warning("🚨 System ML wykrył anomalie czasowe. Przekazywanie zlecenia do Agenta LLM w celu optymalizacji...")
        
        with st.spinner("LLM analizuje scenariusze logistyczne..."):
            wynik_llm = llm_system.generate_mitigation_plan(
                order_id=selected_order_id,
                product_name=order_data['Product Name'],
                price=order_data['Product Price'],
                risk=ryzyko
            )
            
            if wynik_llm and "status_systemowy" in wynik_llm:
                # Wyświetlamy decyzje wygenerowane przez LLM w przejrzystych kafelkach
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.error(f"Status: {wynik_llm.get('status_systemowy', 'N/A')}")
                with c2:
                    st.info(f"Priorytet zgłoszenia: {wynik_llm.get('przypisany_priorytet', 'N/A')}")
                with c3:
                    st.success("Wygenerowano procedurę awaryjną")
                
                # Prezentacja technicznych wniosków
                st.markdown("### 📝 Instrukcja naprawcza (Zalecenie systemowe)")
                st.info(wynik_llm.get('akcja_korekcyjna', 'Brak zalecenia'))
                
                st.markdown("### 🔍 Uzasadnienie biznesowo-logistyczne")
                st.write(wynik_llm.get('uzasadnienie_decyzji', 'Brak uzasadnienia'))
                
                # Dodatkowa sekcja "Surowy JSON" - prowadzący projekty uwielbiają widzieć podgląd API
                with st.expander("🔌 Zobacz surową odpowiedź API (Format JSON)"):
                    st.json(wynik_llm)
            else:
                st.error("Wystąpił błąd parsowania odpowiedzi z lokalnego LLM.")
    else:
        st.success("✅ Parametry zamówienia mieszczą się w normie. Brak konieczności uruchamiania dodatkowych procedur zarządczych.")