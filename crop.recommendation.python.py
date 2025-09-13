# =====================================================
# Step 1: Install dependencies
# =====================================================
!pip install pandas scikit-learn joblib requests xgboost

# =====================================================
# Step 2: Import libraries
# =====================================================
import pandas as pd
import joblib
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# =====================================================
# Step 3: Load dataset (Upload CSV in Colab)
# =====================================================
data = pd.read_csv("/content/Crop_recommendation.csv")

X = data.drop("label", axis=1)
y = data["label"]

# Encode labels (crop names → numbers)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# =====================================================
# Step 4: Train only XGBoost (best model)
# =====================================================
model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=8,
    random_state=42,
    eval_metric='mlogloss'
)

model.fit(X_train, y_train)

# Save trained model + label encoder
joblib.dump(model, "best_crop_model.pkl")
joblib.dump(le, "label_encoder.pkl")

# =====================================================
# Step 5: Soil Type → NPK Mapping
# =====================================================
soil_npk = {
    "clay": (80, 40, 40),
    "loamy": (60, 30, 30),
    "sandy": (50, 20, 20),
    "black": (70, 35, 35)
}

# =====================================================
# Step 6: Weather Function (Optional)
# =====================================================
def get_weather(location, api_key="YOUR_API_KEY"):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"
    response = requests.get(url).json()
    try:
        temp = response['main']['temp']
        humidity = response['main']['humidity']
        rainfall = response.get('rain', {}).get('1h', 50)
        return temp, humidity, rainfall
    except:
        return 25, 60, 100  # fallback if API fails

# =====================================================
# Step 7: Crop Recommendation Function
# =====================================================
def recommend_crop(soil_type, ph, location):
    model = joblib.load("best_crop_model.pkl")
    le = joblib.load("label_encoder.pkl")

    if soil_type not in soil_npk:
        return "❌ Soil type not recognized"

    N, P, K = soil_npk[soil_type]
    temp, humidity, rainfall = get_weather(location)

    features = [[N, P, K, temp, humidity, ph, rainfall]]
    prediction = model.predict(features)

    return le.inverse_transform(prediction)[0]

# =====================================================
# Step 8: Farmer Input
# =====================================================
soil_type = input("Enter soil type (clay/sandy/loamy/black): ").lower()
ph = float(input("Enter soil pH value: "))
location = input("Enter location (city name): ")

recommended_crop = recommend_crop(soil_type, ph, location)

print("\n🌟 Based on soil, pH, and weather conditions...")
print("✅ Recommended Crop for your field:", recommended_crop)
