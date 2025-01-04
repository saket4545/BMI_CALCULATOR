import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
from sklearn.base import BaseEstimator, ClassifierMixin
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class KerasCustomClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, epochs=100, batch_size=32):
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = self.create_model()

    def create_model(self):
        model = Sequential([
            Dense(36, activation='relu', input_shape=(3,)),
            Dense(24, activation='relu'),
            Dense(12, activation='relu'),
            Dense(6, activation='softmax')
        ])
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def fit(self, X, y):
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=1)
        return self

    def predict(self, X):
        return np.argmax(self.model.predict(X), axis=-1)

    def score(self, X, y):
        return self.model.evaluate(X, y, verbose=0)[1]



with open('bmi_pipeline.pkl', 'rb') as file:
    pipeline = pickle.load(file)


categories = {
    0: "Extremely Weak",
    1: "Weak",
    2: "Normal",
    3: "Overweight",
    4: "Obesity",
    5: "Extreme Obesity"
}


st.title("BMI Prediction App")


gender = st.selectbox("Select Gender", ["Male", "Female"])
height = st.number_input("Enter your height (in cm)", min_value=100, max_value=250, value=170)
weight = st.number_input("Enter your weight (in kg)", min_value=30, max_value=250, value=70)


if st.button('Predict BMI Category'):
    
    input_data = pd.DataFrame([[gender, height, weight]], columns=['Gender', 'Height', 'Weight'])

    
    prediction = pipeline.predict(input_data)

  
    category = categories[prediction[0]]

    st.write(f"Your BMI category is: **{category}**")
