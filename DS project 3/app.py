#!/usr/bin/env python
# coding: utf-8

# In[6]:


#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
import pickle
import streamlit as st

data = pd.read_csv("co2_emissions.csv")

make_list = list(data['make'].unique())
model_list = list(data['model'].unique())
vehicle_class_list = list(data['vehicle_class'].unique())
transmission_list = list(data['transmission'].unique())
fuel_type_list = list(data['fuel_type'].unique())

model_filename = 'final_model.pkl'
encoder_filename = 'label_encoders.pkl'
rf_model = pickle.load(open(model_filename, 'rb'))
label_encoders = pickle.load(open(encoder_filename, 'rb'))

                 
def get_predictions(input_data):
    input_df = pd.DataFrame([input_data])
    new_data_point = {
    'make': input_df['make'],
    'model': input_df['model'],
    'vehicle_class': input_df['vehicle_class'],   
    'transmission': input_df['transmission'],
    'fuel_type': input_df['fuel_type']}
    preprocessed_data_point = {}

    # Perform Label Encoding for each categorical column in the new data point
    for column, value in new_data_point.items():
        if column in label_encoders:
            le = label_encoders[column]
            preprocessed_data_point[column] = le.transform([value])[0]
        else:
            preprocessed_data_point[column] = value
    
    input_df['make'] = preprocessed_data_point['make']
    input_df['model'] = preprocessed_data_point['model']
    input_df['vehicle_class'] = preprocessed_data_point['vehicle_class']
    input_df['transmission'] = preprocessed_data_point['transmission']
    input_df['fuel_type'] = preprocessed_data_point['fuel_type']
    
    predictions = rf_model.predict(input_df)
    return predictions

def main():
    st.title('Co2 Emission Predictor')
    st.header('Enter the specifications of the car:')
    make = st.selectbox('Car Make:', make_list)
    model = st.selectbox('Car Model:', model_list)
    vehicle_class = st.selectbox('Vehicle Class:', vehicle_class_list)
    engine_size = st.number_input('Engine Size:')
    cylinders = st.number_input('No of Cylinders:')
    transmission = st.selectbox('Transmission Type:', transmission_list)
    fuel_type = st.selectbox('Fuel Type:', fuel_type_list)
    fuel_consumption_city = st.number_input('Fuel consumption in city:')
    fuel_consumption_hwy = st.number_input('Fuel consumption in highway:')
    fuel_consumption_comb = st.number_input('Fuel consumption combusion(Litres per 100km):')
    fuel_consumption_comb_mpg = st.number_input('Fuel consumption combusion(Mile per Gallon):')
    
    
    input_data = {'make': make, 'model': model, 'vehicle_class': vehicle_class, 'engine_size':engine_size, 'cylinders': cylinders, 'transmission': transmission, 'fuel_type': fuel_type, 'fuel_consumption_city': fuel_consumption_city, 'fuel_consumption_hwy': fuel_consumption_hwy, 'fuel_consumption_comb(l/100km)': fuel_consumption_comb, 'fuel_consumption_comb(mpg)': fuel_consumption_comb_mpg}

    if st.button('Predict co2 emission'):
        co2_emission = get_predictions(input_data)
        st.success(f'The predicted co2 emission for the given car is {co2_emission[0]}')


if __name__ == '__main__':
    main()





# In[ ]:




