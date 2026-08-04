import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

class SupplyChainML:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.features = [
            'Days for shipment (scheduled)',
            'Benefit per order',
            'Sales per customer',
            'Product Price'
        ]

    def train(self, df):
        X = df[self.features].fillna(df[self.features].median())
        y = df['Late_delivery_risk']
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
        print("Model ML został pomyślnie wytrenowany.")

    def predict_risk(self, product_data):
        input_df = pd.DataFrame([{
            'Days for shipment (scheduled)': product_data['Days for shipment (scheduled)'].median(),
            'Benefit per order': product_data['Benefit per order'].median(),
            'Sales per customer': product_data['Sales per customer'].median(),
            'Product Price': product_data['Product Price'].median()
        }])
        scaled_input = self.scaler.transform(input_df)
        probability = self.model.predict_proba(scaled_input)[0][1]
        return probability
