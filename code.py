# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 11:09:16 2022

@author: Hilman

Project 2
Productivity Production of Garment Employees Dataset
"""

#1. Import the packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.callbacks import EarlyStopping, TensorBoard
import datetime, os
import pandas as pd

#2. Import and read dataset 
data_path = r"C:\Users\Ryzen\Documents\tensorflow\data\garments_worker_productivity.csv"
df = pd.read_csv(data_path)

#%%
#Inspect is there any NA value
print(df.isna().sum())

#%%
#3. Data cleaning
wip_mean_value = df['wip'].mean()

#Fill NA values with mean value of the column
gworker = df.copy()
gworker['wip'].fillna(value=wip_mean_value, inplace=True)

print(gworker.isna().sum())
#%%
#4. Data preprocessing
#Change 'sweing' to 'sewing'
gworker['department'] = gworker['department'].replace(['sweing'],['sewing'])
#Change 'finishing ' to 'finishing'
gworker['department'] = gworker['department'].replace(['finishing '], ['finishing'])

#Change date from object to datetime
gworker['date'] = pd.to_datetime(gworker['date'], format='%m/%d/%Y').dt.date

gworker = gworker.drop(['date','idle_time','idle_men','no_of_style_change'], axis=1)
#%%
# Use LabelEncoder to quarter, department, and day column
class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)
    
MultiColumnLabelEncoder(columns = ['quarter', 'department', 'day']).fit_transform(gworker)
MultiColumnLabelEncoder().fit_transform(gworker.drop(['quarter', 'department', 'day'],axis=1))

#%%
#5. Split data into features and labels
gworker = pd.get_dummies(gworker)
features = gworker.copy()
labels = features.pop('actual_productivity')

#%%
#6. Do a train-test split
SEED=12345
x_train, x_test, y_train, y_test = train_test_split(features,labels,test_size=0.2, random_state=SEED)

#7. Perform data normalization
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# Data preparation is completed here...

#%%
#8. Build NN model
nIn= x_train.shape[-1]
nClass = y_train.shape[-1]

# Use functional API
inputs = keras.Input(shape=(nIn,))
h1 = layers.Dense(4096, activation='linear')
h2 = layers.Dense(512, activation='linear')
h3 = layers.Dense(2048, activation='linear')
h4 = layers.Dense(64, activation='linear')
out = layers.Dense(1)

x = h1(inputs)
x = h2(x)
x = h3(x)
x = h4(x)
outputs = out(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()

#Use in iPython console 
tf.keras.utils.plot_model(model, show_shapes=True)

#%%
#9. Compile and train the model
base_log_path = r"C:\Users\Ryzen\Documents\tensorflow\GitHub\Project-2-Productivity-Production-of-Garment-Employees-Dataset\tb_logs"
log_path= os.path.join(base_log_path, datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '__Project_2')
tb = TensorBoard(log_dir=log_path)
es = EarlyStopping(monitor='val_loss', patience=200, verbose=2)

model.compile(optimizer='adam', loss='mse', metrics=['mae','mse'])
model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=128, epochs=1000, callbacks=[tb,es])

#%%
#10. Evaluate with test data for wild testing
test_result = model.evaluate(x_test,y_test,batch_size=128)
print(f"Test loss = {test_result[0]}")
print(f"Test MAE = {test_result[1]}")
print(f"Test MSE = {test_result[2]}")

#%%
import matplotlib.pyplot as plt
#11. Plot a graph of prediction vs label on test data
predictions = np.squeeze(model.predict(x_test))
labels = np.squeeze(y_test)
plt.plot(predictions,labels,".")
plt.xlabel("Predictions")
plt.ylabel("Labels")
plt.title("Graph of Predictions vs Labels with Test Data")
save_path = r"C:\Users\Ryzen\Documents\tensorflow\GitHub\Project-2-Productivity-Production-of-Garment-Employees-Dataset"
plt.savefig(os.path.join(save_path,"result.png"),bbox_inches='tight')
plt.show()
#%%
from numba import cuda 
device = cuda.get_current_device()
device.reset()