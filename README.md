# Productivity Prediction of Garment Employees Using Feedforward Neural Network

## Hardware Specifications
AMD Ryzen 7 3700x

NVIDIA Geforce RTX 3050

32GB 3200MHZ DDR4

## 1. Summary
The goal of this project is to predict the actual productivity of garment employees, based on the following features:
- quarter year
- department
- day

The dataset is obtained from [here](https://www.kaggle.com/datasets/ishadss/productivity-prediction-of-garment-employees).

## 2. IDE and Framework
This project is created using Sypder 5.3.1 as the main IDE. The main frameworks used in this project are Pandas, Numpy, Scikit-learn and TensorFlow Keras.

## 3. Methodology
### 3.1. Data Pipeline
The data is first loaded and preprocessed, such that unwanted features are removed. Categorical features are label encode. Then the data is split into train-validation-test sets, with a ratio of 60:20:20.

### 3.2. Model Pipeline
A feedforward neural network is constructed that is catered for regression problem. The structure of the model is fairly simple. Figure below shows the structure of the model.

![Model](img/model.png)

The model is trained with a batch size of 128 and for 1000 epochs. Early stopping is applied in this training. The training stops at epoch 321, with a training MAE of 0.1141 and validation MAE of 0.1077. The two figures below show the graph of the training process, indicating the convergence of model training.

![Loss Graph](img/loss.PNG) ![MAE Graph](img/mae.PNG)

## 4. Results
The model are tested with test data. The evaluation result is shown in figure below.

![Test Result](img/test_result.PNG)

The model is also used to made prediction with test data. A graph of prediction vs label is plotted, as shown in the image below.

![Graph Result](img/result.png)

Based on the graph, a clear trendline of y=x can be seen, indicating the predictions are fairly similar as labels. However, several outliers can be seen in the graph.