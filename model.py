"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json

# Additional libraries for data preprocessing
from sklearn.preprocessing import StandardScaler
from holidays import Spain


def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.
    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------
    # predict_vector = feature_vector_df[['Madrid_wind_speed','Bilbao_rain_1h','Valencia_wind_speed']]

    # deal with missing values
    def data_cleaning(data_table):
        df = data_table.drop("time", axis=1)  # Get all features excluding "time"
        
        df = df.drop("Unnamed: 0", axis=1)  # drop "Unnamed: 0"
        
        # DATA CLEANING
        df["Valencia_wind_deg"] = df["Valencia_wind_deg"].str.strip().str.replace("level_", "")
        df["Valencia_wind_deg"] = df["Valencia_wind_deg"].astype(float)

        df["Seville_pressure"] = df["Seville_pressure"].str.strip().str.replace("sp", "")
        df["Seville_pressure"] = df["Seville_pressure"].astype(float)

        vp_mean = round(df["Valencia_pressure"].mean(), 6)

        df["Valencia_pressure"] = df["Valencia_pressure"].fillna(vp_mean)
        
        df = df.astype(float)  # Convert other features to float datatype

        df['date'] = pd.to_datetime(data_table['time']) # --------------------
        df["time"] = data_table["time"]  # get the time column
        
        df = df.fillna(0)  # fill future unknown null values from real world dataset

        return df

    new_train = data_cleaning(feature_vector_df)

    # create new features
    def add_features(clean_data):
        df = clean_data.copy()
        
        # Create a new holiday calendar for Spain
        cal = Spain()
        # Create a new column 'is_holiday' based on whether the date is a holiday in Spain
        df['is_holiday'] = df['date'].apply(lambda x: x in cal).astype(int)

        # define a dictionary of month numbers and season labels
        seasons = {1: 'winter', 2: 'winter', 3: 'spring', 
            4: 'spring', 5: 'spring', 6: 'summer', 
            7: 'summer', 8: 'summer', 9: 'autumn', 
            10: 'autumn', 11: 'autumn', 12: 'winter'}
        # use the map() method to apply the dictionary to the month number
        df['season'] = df['date'].dt.month.map(seasons)

        # converting season to categorical feature
        temp = {"winter": 1, "spring": 2, "summer": 3, "autumn": 4}
        df['season'] = df['season'].map(temp)

        # create 'is_hot' column
        df['is_hot'] = df['season'].map(lambda season: int(season == 3))

        # create 'is_cold' column
        df['is_cold'] = df['season'].map(lambda season: int(season == 1))

        # create 'day_of_week' column
        df['day_of_week'] = df['date'].dt.weekday + 1
        
        df["time"] = df["time"].str.replace("-","").str.replace(":","").str.replace(" ","")

        df["month"] = df["time"].str[4:6].astype(int) 
        
        df["day"] = df["time"].str[6:8].astype(int)

        df["hour"] = df["time"].str[8:10].astype(int)
        
        df = df.drop(["time", "date", "season"], axis=1)
        
        return df

    df_train_new = add_features(new_train)

    # engineer existing features
    def feature_engr(data):
        new_df = data.copy()

        # remove "Seville_pressure" and "Barcelona_pressure" features
        new_df = new_df.drop(["Seville_pressure", "Barcelona_pressure"], axis=1)

        new_pressure_cols = [col for col in new_df.columns if "pressure" in col]
        new_df["total_pressure_mean"] = new_df[new_pressure_cols].mean(axis=1)

        cities = ["Seville", "Bilbao", "Madrid", "Valencia", "Barcelona"]
        for city in cities:   
            # Adding the Temperature mean values for each city 
            temp_list = [f"{city}_temp", f"{city}_temp_max", f"{city}_temp_min"]
            new_df[f"{city}_temp_mean"] = new_df[temp_list].mean(axis=1)

        humidity = [column for column in new_df.columns if "humidity" in column]
        # Aggregate the humidity features
        new_df["humidity_mean"] = new_df[humidity].mean(axis=1)

        # dropping "Valencia_wind_speed"
        new_df = new_df.drop("Valencia_wind_speed", axis=1)

        # updated wind_speed features
        wind_speed = [column for column in new_df.columns if "wind_speed" in column]

        # Aggregate the wind_speed features
        new_df["wind_speed_mean"] = new_df[wind_speed].mean(axis=1)
        
        wind_deg = [column for column in new_df.columns if "wind_deg" in column]
        new_df["wind_deg_mean"] = new_df[wind_deg].mean(axis=1)
        
        clouds_all = [column for column in new_df.columns if "cloud" in column]
        new_df["clouds_all_mean"] = new_df[clouds_all].mean(axis=1)

        # # move column 'load_shortfall_3h' to the right edge
        if "load_shortfall_3h" in new_df.columns:
            new_df = new_df.reindex(
                columns=[col for col in new_df.columns if col != 'load_shortfall_3h'] + 
                ['load_shortfall_3h'])
        
        return new_df

    std_train = feature_engr(df_train_new)

    # Scale the data
    def standardiser(dataset):

        df = dataset.copy()

        # select columns to standardize
        features = ["Valencia", "Madrid", "Seville", "Bilbao", "Barcelona", "snow", "rain"]
        cols_to_standardize = [var for var in df.columns for col in features if col in var]

        # create a StandardScaler object
        scaler = StandardScaler()

        # fit the scaler to the selected columns and transform them
        df[cols_to_standardize] = scaler.fit_transform(df[cols_to_standardize])

        return df

    feature_vector_df = standardiser(std_train)


    predict_vector = feature_vector_df[
        ['Madrid_wind_speed', 'Valencia_wind_deg', 'Bilbao_rain_1h',
       'Seville_humidity', 'Madrid_humidity', 'Bilbao_clouds_all',
       'Bilbao_wind_speed', 'Seville_clouds_all', 'Bilbao_wind_deg',
       'Barcelona_wind_speed', 'Barcelona_wind_deg', 'Madrid_clouds_all',
       'Seville_wind_speed', 'Barcelona_rain_1h', 'Seville_rain_1h',
       'Bilbao_snow_3h', 'Seville_rain_3h', 'Madrid_rain_1h',
       'Barcelona_rain_3h', 'Valencia_snow_3h', 'Madrid_weather_id',
       'Barcelona_weather_id', 'Bilbao_pressure', 'Seville_weather_id',
       'Valencia_pressure', 'Seville_temp_max', 'Madrid_pressure',
       'Valencia_temp_max', 'Valencia_temp', 'Bilbao_weather_id',
       'Seville_temp', 'Valencia_humidity', 'Valencia_temp_min',
       'Barcelona_temp_max', 'Madrid_temp_max', 'Barcelona_temp',
       'Bilbao_temp_min', 'Bilbao_temp', 'Barcelona_temp_min',
       'Bilbao_temp_max', 'Seville_temp_min', 'Madrid_temp', 'Madrid_temp_min',
       'is_holiday', 'is_hot', 'is_cold', 'day_of_week', 'month', 'day',
       'hour', 'total_pressure_mean', 'Seville_temp_mean', 'Bilbao_temp_mean',
       'Madrid_temp_mean', 'Valencia_temp_mean', 'Barcelona_temp_mean',
       'humidity_mean', 'wind_speed_mean', 'wind_deg_mean', 'clouds_all_mean']
    ]
    # ------------------------------------------------------------------------

    return predict_vector

    

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))


""" You may use this section (above the make_prediction function) of the python script to implement 
    any auxiliary functions required to process your model's artifacts.
"""

def make_prediction(data, model):
    """Prepare request data for model prediction.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standardisation.
    return prediction[0].tolist()
