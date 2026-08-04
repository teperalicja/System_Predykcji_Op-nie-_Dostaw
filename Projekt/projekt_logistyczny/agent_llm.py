import requests
import json

class SupplyChainLLM:
    def __init__(self, model_name="llama3"):
        self.url = "http://localhost:11434/api/generate"
        self.model_name = model_name

    def generate_mitigation_plan(self, product_name, price, risk):
        prompt = f"""
        Jesteś menedżerem ds. logistyki. System ML wykrył ryzyko opóźnienia dostawy ({risk:.0%}) dla produktu: {product_name} (Cena: {price} USD).
        
        Zwróć odpowiedź WYŁĄCZNIE jako czysty obiekt JSON o podanej strukturze (nie pisz żadnego innego tekstu!):
        {{
            "status": "alert",
            "rekomendacja_logistyczna": "Jedno konkretne działanie naprawcze dla magazynu po polsku.",
            "email_klient": "Krótki, profesjonalny i uprzejmy e-mail do klienta po polsku informujący o możliwym opóźnieniu i dający kod rabatowy REK15."
        }}
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }
        try:
            response = requests.post(self.url, json=payload, timeout=15)
            if response.status_code == 200:
                return json.loads(response.json()['response'])
        except Exception as e:
            return {"status": "error", "message": f"Błąd połączenia z LLM: {e}"}
