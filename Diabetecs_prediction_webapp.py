import numpy as np
import pickle
import streamlit as st

#load the save model
loaded_model = pickle.load(open('C:/Users/Tharanga Mawan/Documents/ML Projects/diabetecs_model.sav', 'rb')) 



'''def standard_normalization(data):
  mean = np.mean(data)
  std = np.std(data)
  scaled_data = (data - mean) / std
  return scaled_data'''

# creaing a function for prediction

def diabetes_prediction(input_data):
    
    # Convert input data to floats and handle non-numeric inputs
    try:
        input_data = [float(value) for value in input_data]
    except ValueError:
        return "Invalid input. Please enter numeric values."
    
    # making numpy array
    data_as_array = np.asarray(input_data)


    input_data_reshaped = data_as_array.reshape(1,-1)

    #standerize input data

    std_data = standard_normalization(input_data_reshaped)
    

    prediction = loaded_model.predict(std_data)
    print(prediction)

    if prediction[0] == 1:
        return "The person is diabetic"

    else:
        return "The person is not diabetic"

def standard_normalization(data):
  mean = np.mean(data)
  std = np.std(data)
  scaled_data = (data - mean) / std
  return scaled_data


def main():


    # giving a title
    st.title("Diabetes Prediction webapp")

    #getting the data inputs

    Pregnancies = st.text_input("Number of pregnancies")
    Glucose = st.text_input("Glucose level")
    BloodPressure = st.text_input("Blood pressure value")
    SkinThickness = st.text_input("Skin Thickness value")
    Insulin = st.text_input("Insulin Level")
    BMI = st.text_input("BMI value")
    DiabetesPedigreeFunction = st.text_input("Diabetes prediction function value")
    Age = st.text_input("Age of the person")


    #code for prediction
    diagnosis = ''

    #creating a button for prediction

    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])

        st.success(diagnosis)


if __name__ == '__main__':
    main()