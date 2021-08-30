import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow.keras
from keras.models import Sequential
from keras.layers import Dense


def load_dataset(dataset):
    car_df = pd.read_csv(dataset)
    logging.info(car_df.head(5))
    return car_df


def visualise_dataset():
    sns.pairplot(load_dataset("../1_car_sales_prediction_tutorial/P74-Project-1/Car_Purchasing_Data.csv"))


def separate_and_clean_data(car_df):
    X = car_df.drop(['Customer Name', 'Customer e-mail', 'Country', 'Car Purchase Amount'], axis=1)
    y = car_df['Car Purchase Amount']
    return X, y


def scale_data(car_df):
    X, y = separate_and_clean_data(load_dataset(car_df))
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    y = y.values.reshape(-1, 1)
    y_scaled = scaler.fit_transform(y)
    return X_scaled, y_scaled


def train_model(car_df):
    X_scaled, y_scaled = scale_data(car_df)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.20)
    model = Sequential()
    model.add(Dense(25, input_dim = 5, activation='relu'))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(1, activation = 'linear'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    epochs_hist = model.fit(X_train, y_train, epochs=10, batch_size=50, verbose=1, validation_split=0.2)
    logging.info(model.summary())
    return model, epochs_hist


def plot_results(epochs_hist):
    plt.plot(epochs_hist.history['loss'])
    plt.plot(epochs_hist.history['val_loss'])

    plt.title('Model Loss Progression During Training/Validation')
    plt.ylabel('Training and Validation Losses')
    plt.xlabel('Epoch Number')
    plt.legend(['Training Loss', 'Validation Loss'])


def make_a_prediction(car_df):
    model = train_model(car_df)[0]
    # Gender, Age, Annual Salary, Credit Card Debt, Net Worth
    X_Testing = np.array([[1, 50, 50000, 10985, 629312]])
    y_predict = model.predict(X_Testing)
    logging.info('Expected Purchase Amount=', y_predict[:, 0])
