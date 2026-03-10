import streamlit as st
import pandas as pd
import pickle

# Load model
model = pickle.load(open("model_diamond_price.pkl", "rb"))
feature_columns = pickle.load(open("feature_columns.pkl", "rb"))

st.title("💎 Diamond Price Prediction")
st.write("Prediksi harga berlian menggunakan Machine Learning")

# INPUT USER
carat = st.number_input("Carat", 0.0, 5.0, 1.0)
depth = st.number_input("Depth", 40.0, 80.0, 60.0)
table = st.number_input("Table", 40.0, 80.0, 55.0)
x = st.number_input("Length (x)", 0.0, 10.0, 5.0)
y = st.number_input("Width (y)", 0.0, 10.0, 5.0)
z = st.number_input("Depth (z)", 0.0, 10.0, 3.0)

cut = st.selectbox("Cut", ["Fair","Good","Very Good","Premium","Ideal"])
color = st.selectbox("Color", ["D","E","F","G","H","I","J"])
clarity = st.selectbox("Clarity", ["I1","SI2","SI1","VS2","VS1","VVS2","VVS1","IF"])

# Data input
input_dict = {
    "carat":carat,
    "depth":depth,
    "table":table,
    "x":x,
    "y":y,
    "z":z,
    "cut":cut,
    "color":color,
    "clarity":clarity
}

input_df = pd.DataFrame([input_dict])

# Encoding kategori
input_encoded = pd.get_dummies(input_df)

input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

# Prediksi
if st.button("Predict Price"):
    prediction = model.predict(input_encoded)

    st.success(f"Predicted Price: ${prediction[0]:,.2f}")
