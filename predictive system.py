# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle

# loading the saved model
loaded_model = pickle.load(open(r'C:\Users\Akshit Singh Rawat\Desktop\pbl\trained_model.sav', 'rb'))

# input data for testing
input_data = (6, 148, 72, 35, 0, 33.6, 0.627, 50)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# make prediction
prediction = loaded_model.predict(input_data_reshaped)
print("Raw prediction output:", prediction)

# show readable result
if prediction[0] == 0:
    print('The person is not diabetic')
else:
    print('The person is diabetic')
