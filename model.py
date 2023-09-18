import numpy as np
import pickle 
from sklearn.preprocessing import StandardScaler

def standard_normalization(data):
  mean = np.mean(data)
  std = np.std(data)
  scaled_data = (data - mean) / std
  return scaled_data



#load the save model
loaded_model = pickle.load(open('C:/Users/Tharanga Mawan/Documents/ML Projects/diabetecs_model.sav', 'rb')) 



input_data = (7,62,78,0,0,32.6,0.391,41)

# making numpy array
data_as_array = np.asarray(input_data)


input_data_reshaped = data_as_array.reshape(1,-1)

#standerize input data

std_data = standard_normalization(input_data_reshaped)
print(std_data)

prediction = loaded_model.predict(std_data)
print(prediction)

if prediction == 1:
    print("The person is diabetic")

else:
    print("The person is not diabetic")  

