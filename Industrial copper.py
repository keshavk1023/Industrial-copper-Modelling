import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelBinarizer
import streamlit as st
import re
import pickle

st.set_page_config(layout="wide")

st.write("""
<div style='text-align:center'>
    <h1 style='color:#009999;'>Industrial Copper Modeling Application</h1>
</div>
""", unsafe_allow_html=True)

# Load models and scalers once
with open(r"C:\\Users\\jeetg\\code\\copper modelling\\model.pkl", 'rb') as file:
    loaded_model = pickle.load(file)
with open(r'C:\\Users\\jeetg\\code\\copper modelling\\scaler.pkl', 'rb') as f:
    scaler_loaded = pickle.load(f)
with open(r"C:\\Users\\jeetg\\code\\copper modelling\\t.pkl", 'rb') as f:
    t_loaded = pickle.load(f)
with open(r"C:\\Users\\jeetg\\code\\copper modelling\\s.pkl", 'rb') as f:
    s_loaded = pickle.load(f)
with open(r"C:\\Users\\jeetg\\code\\copper modelling\\cmodel.pkl", 'rb') as file:
    cloaded_model = pickle.load(file)
with open(r'C:\\Users\\jeetg\\code\\copper modelling\\cscaler.pkl', 'rb') as f:
    cscaler_loaded = pickle.load(f)
with open(r"C:\\Users\\jeetg\\code\\copper modelling\\ct.pkl", 'rb') as f:
    ct_loaded = pickle.load(f)

# Define possible values for dropdown menus
status_options = ['Won', 'Draft', 'To be approved', 'Lost', 'Not lost for AM', 'Wonderful', 'Revised', 'Offered', 'Offerable']
item_type_options = ['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR']
country_options = [28., 25., 30., 32., 38., 78., 27., 77., 113., 79., 26., 39., 40., 84., 80., 107., 89.]
application_options = [10., 41., 28., 59., 15., 4., 38., 56., 42., 26., 27., 19., 20., 66., 29., 22., 40., 25., 67., 79., 3., 99., 2., 5., 39., 69., 70., 65., 58., 68.]
product = ['611112', '611728', '628112', '628117', '628377', '640400', '640405', '640665', '611993', '929423819', '1282007633', '1332077137', '164141591', '164336407', '164337175', '1665572032', '1665572374', '1665584320', '1665584642', '1665584662', '1668701376', '1668701698', '1668701718', '1668701725', '1670798778', '1671863738', '1671876026', '1690738206', '1690738219', '1693867550', '1693867563', '1721130331', '1722207579']

def validate_input(inputs):
    pattern = "^(?:\d+|\d*\.\d+)$"
    for input_value in inputs:
        if not re.match(pattern, input_value):
            return False, input_value
    return True, None

def preprocess_and_predict_sell_price(quantity_tons, application, thickness, width, country, customer, product_ref, item_type, status):
    new_sample = np.array([[np.log(float(quantity_tons)), application, np.log(float(thickness)), float(width), country, float(customer), int(product_ref), item_type, status]])
    new_sample_ohe = t_loaded.transform(new_sample[:, [7]]).toarray()
    new_sample_be = s_loaded.transform(new_sample[:, [8]]).toarray()
    new_sample = np.concatenate((new_sample[:, [0, 1, 2, 3, 4, 5, 6]], new_sample_ohe, new_sample_be), axis=1)
    new_sample = scaler_loaded.transform(new_sample)
    new_pred = loaded_model.predict(new_sample)[0]
    return np.exp(new_pred)

def preprocess_and_predict_status(cquantity_tons, cselling, capplication, cthickness, cwidth, ccountry, ccustomer, cproduct_ref, citem_type):
    new_sample = np.array([[np.log(float(cquantity_tons)), np.log(float(cselling)), capplication, np.log(float(cthickness)), float(cwidth), ccountry, int(ccustomer), int(cproduct_ref), citem_type]])
    new_sample_ohe = ct_loaded.transform(new_sample[:, [8]]).toarray()
    new_sample = np.concatenate((new_sample[:, [0, 1, 2, 3, 4, 5, 6, 7]], new_sample_ohe), axis=1)
    new_sample = cscaler_loaded.transform(new_sample)
    new_pred = cloaded_model.predict(new_sample)
    return 'Won' if new_pred == 1 else 'Lost'

tab1, tab2 = st.tabs(["PREDICT SELLING PRICE", "PREDICT STATUS"])

with tab1:
    with st.form("my_form"):
        col1, col3 = st.columns([5, 5])
        with col1:
            status = st.selectbox("Status", status_options, key=1)
            item_type = st.selectbox("Item Type", item_type_options, key=2)
            country = st.selectbox("Country", sorted(country_options), key=3)
            application = st.selectbox("Application", sorted(application_options), key=4)
            product_ref = st.selectbox("Product Reference", product, key=5)
        with col3:
            st.write('<h5 style="color:rgb(0, 153, 153,0.4);">NOTE: Min & Max given for reference, you can enter any value</h5>', unsafe_allow_html=True)
            quantity_tons = st.text_input("Enter Quantity Tons (Min:611728 & Max:1722207579)")
            thickness = st.text_input("Enter thickness (Min:0.18 & Max:400)")
            width = st.text_input("Enter width (Min:1, Max:2990)")
            customer = st.text_input("customer ID (Min:12458, Max:30408185)")
            submit_button = st.form_submit_button(label="PREDICT SELLING PRICE")
            st.markdown("""
                <style>
                div.stButton > button:first-child {
                    background-color: #009999;
                    color: white;
                    width: 100%;
                }
                </style>
            """, unsafe_allow_html=True)

    if submit_button:
        is_valid, invalid_value = validate_input([quantity_tons, thickness, width, customer])
        if not is_valid:
            st.write(f"Invalid input: {invalid_value}. Please enter a valid number.")
        else:
            predicted_price = preprocess_and_predict_sell_price(quantity_tons, application, thickness, width, country, customer, product_ref, item_type, status)
            st.write('## :green[Predicted selling price:] ', predicted_price)

with tab2:
    with st.form("my_form1"):
        col1, col3 = st.columns([5, 5])
        with col1:
            cquantity_tons = st.text_input("Enter Quantity Tons (Min:611728 & Max:1722207579)")
            cthickness = st.text_input("Enter thickness (Min:0.18 & Max:400)")
            cwidth = st.text_input("Enter width (Min:1, Max:2990)")
            ccustomer = st.text_input("customer ID (Min:12458, Max:30408185)")
            cselling = st.text_input("Selling Price (Min:1, Max:100001015)")
        with col3:
            citem_type = st.selectbox("Item Type", item_type_options, key=21)
            ccountry = st.selectbox("Country", sorted(country_options), key=31)
            capplication = st.selectbox("Application", sorted(application_options), key=41)
            cproduct_ref = st.selectbox("Product Reference", product, key=51)
            csubmit_button = st.form_submit_button(label="PREDICT STATUS")

    if csubmit_button:
        is_valid, invalid_value = validate_input([cquantity_tons, cthickness, cwidth, ccustomer, cselling])
        if not is_valid:
            st.write(f"Invalid input: {invalid_value}. Please enter a valid number.")
        else:
            predicted_status = preprocess_and_predict_status(cquantity_tons, cselling, capplication, cthickness, cwidth, ccountry, ccustomer, cproduct_ref, citem_type)
            status_color = 'green' if predicted_status == 'Won' else 'red'
            st.write(f'## :{status_color}[The Status is {predicted_status}] ')