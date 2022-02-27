import pandas as pd
import numpy as np
import datetime as datetime
import streamlit as st
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import joblib
import time
import pickle

def main():
    st.title('ASHRAE - Great Energy Predictor III')
    st.markdown('Just Enter the following details and we will **predict metered building energy usage**')
    st.sidebar.title("Prepared as project for Diploma in Artificial Intelligence & Machine Learning at UoH and AAIC")
    st.sidebar.markdown('Submitted by [[Saurav Kumar Sahay]'
                        '(https://www.linkedin.com/in/sahaysaurav/)]')
    st.warning('All * are mandotory to choose, for others if no value is provided default values will be picked')
    df = get_data()
    if st.checkbox('Show data I have entered'):
        st.write(df)
    df1 = preprocess_data(df)
    if st.checkbox('Show Preprocessed data'):
        st.write(df1)
    csv = convert_df(df1)
    st.download_button("Press to Download",csv,"file.csv","text/csv",key='download-csv')
        #df1.to_csv("coverted.csv")
    submit = st.button('Predict Energy Consumption')
    model = joblib.load('decision_tree_reg_modified.sav')
    if submit:
        try:
            with st.spinner('Predicting...'):
                time.sleep(2)
                predictions_log = model.predict(df1)
                predictions = 10**predictions_log-1
                answer = str(round(predictions[0], 2))
                st.info(f"Your **predict metered building energy usage** is **{answer}** Units")
                #st.write(predictions)
        except:
            st.warning("Time value is missing kindly recheck your data")
def get_data():
    building_id = st.slider("* A number between 1-1449", min_value = 1, max_value = 1449)
    meter = st.selectbox("* Meter Type 0: electricity, 1: chilledwater, 2: steam, 3: hotwater",['0', '1', '2', '3'])
    timestamp = st.text_input("* Inter timestamp valid type YYYY-MM-DD HH:MM:SS, eg ' 2017-12-31 01:00:00'")
    site_id = st.slider("A number between 0-15", min_value = 0, max_value = 15)
    primary_use = st.selectbox("Chosse the primary use for building",['Retail', 'Office', 'Education', 'Entertainment/public assembly','Warehouse/storage', 'Services', 'Lodging/residential', 'Parking',
       'Public services', 'Healthcare', 'Manufacturing/industrial','Other', 'Technology/science', 'Food sales and service', 'Utility',
       'Religious worship'])
    square_feet = st.slider("Area in square feets", min_value = 2000, max_value = 100000,step = 100, value = 72700)
    year_built = st.slider("Year in which the building was built", min_value = 1900, max_value = 2020, value = 1970)
    floor_count = st.slider("Number of floors in the building",min_value = 1, max_value = 30, value = 3)
    air_temperature = st.slider("Air temperature in degree celcious", min_value = -30.0, max_value = 50.0, step = 0.5, value = 17.0)
    cloud_coverage = st.slider("Select the cloud coverage", min_value = 0.0, max_value = 10.0, step = 0.5, value = 0.0)
    dew_temperature = st.slider("Dew temperature in degree celcious", min_value = -4.0, max_value = 27.0, step = 0.5, value = 9.0)
    precip_depth_1_hr = st.slider("precipitation depth at one hour ", min_value = -1.0, max_value = 30.0, step = 0.5, value = 0.0)
    sea_level_pressure = st.slider("Sea level pressure in Bar", min_value = 900, max_value = 1070, value = 1015)
    wind_direction = st.slider("Direction of wind in 360 degree", min_value = 0, max_value = 360, value = 180)
    wind_speed = st.slider("Speed of wind in meter per seconds", min_value=0.0, max_value= 30.0, value = 3.1, step = 0.1 )
    x = {"building_id" : [building_id],"meter" : [meter],"timestamp"  : [timestamp],"site_id":[site_id], "primary_use":[primary_use],"square_feet":[square_feet],"year_built":[year_built],
    "floor_count":[floor_count],"air_temperature":[air_temperature],"cloud_coverage"  : [cloud_coverage],"dew_temperature" : [dew_temperature],"precip_depth_1_hr":[precip_depth_1_hr],
    "sea_level_pressure":[sea_level_pressure],"wind_direction":[wind_direction],"wind_speed":[wind_speed]}        
    df = pd.DataFrame.from_dict(x)  
    return df
def preprocess_data(df):
    df["timestamp"] = pd.to_datetime(df["timestamp"]) # Converting object type to datetime format for further analysis
    df["hour"] = df["timestamp"].dt.hour # extrating hour from date time and creating new Varibale
    df["DayofWeek"] = df["timestamp"].dt.dayofweek # extrating day of week from date time and creating new Varibale, Monday 0
    df["Month"] = df["timestamp"].dt.month # extrating month from date time and creating new Varibale
    # Creating new features
    df["Weekend"] = df["DayofWeek"] > 4
    df['WorkingHour']= df['timestamp'].apply(lambda x: 1 if x.hour >=6 and x.hour <=20 else 0)
    df['relative_humidity']= 100*((np.exp((17.625*df['dew_temperature'])/
                                            (243.04+df['dew_temperature'])))/(np.exp((17.625*df['air_temperature'])/
                                                                                          (243.04+df['air_temperature']))))
    df.drop(['year_built', 'floor_count','site_id', 'dew_temperature','timestamp',"primary_use"], axis=1,inplace=True) # Droping year built and floor_count
    return df
def convert_df(df):
   return df.to_csv().encode('utf-8')

   
if __name__ == '__main__':
    main()
    
