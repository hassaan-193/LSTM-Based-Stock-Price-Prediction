import streamlit as st
import pickle
import numpy as np
import datetime

# Load the trained model
with open('Car_Price_Predicting_System.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("🚗 Car Price Prediction System")
st.write("Enter the details of the car to predict its selling price.")

# Input fields
present_price = st.number_input("Present Price (in lakhs)", value=5.0)
kms_driven = st.number_input("Kilometers Driven", value=50000)
owner = st.selectbox("Number of Previous Owners", [0, 1, 2, 3])
year = st.number_input("Year of Purchase", min_value=1990, max_value=datetime.datetime.now().year, value=2015)

fuel_type = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG'])
seller_type = st.selectbox("Seller Type", ['Dealer', 'Individual'])
transmission = st.selectbox("Transmission Type", ['Manual', 'Automatic'])

# Manual one-hot encoding to match training data
fuel_diesel = 1 if fuel_type == 'Diesel' else 0
fuel_petrol = 1 if fuel_type == 'Petrol' else 0
fuel_cng = 1 if fuel_type == 'CNG' else 0

seller_dealer = 1 if seller_type == 'Dealer' else 0
seller_individual = 1 if seller_type == 'Individual' else 0

trans_manual = 1 if transmission == 'Manual' else 0
trans_auto = 1 if transmission == 'Automatic' else 0

# If your model had additional dummy variables during training, set them to 0
# Replace or remove as needed
extra1 = 0
extra2 = 0

# Final feature vector (13 features)
input_data = np.array([[present_price, kms_driven, owner, year,
                        fuel_diesel, fuel_petrol, fuel_cng,
                        seller_dealer, seller_individual,
                        trans_manual, trans_auto,
                        extra1, extra2]])

# Prediction button
if st.button("Predict Price"):
    prediction = model.predict(input_data)
    st.success(f"Estimated Selling Price: ₹ {round(prediction[0], 2)} lakhs")