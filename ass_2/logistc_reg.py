import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def standard_scalar(data):
    mean = np.mean(data[:, 0:4], axis=0).reshape((1, 4))
    data[:, 0:4] = data[:, 0:4] - mean
    std = np.std(data[:, 0:4], axis=0).reshape((1, 4))
    data[:, 0:4] = data[:, 0:4]/std
    return data


def min_max_scalar(data):
    data[:, 0:4] = (data[:, 0:4] - np.amin(data[:, 0:4], axis=0)) / (
                np.amax(data[:, 0:4], axis=0) - np.amin(data[:, 0:4], axis=0))
    return data


def scores(X, y, w, threshold=0.5):
    y_pred = X.dot(w)
    class_1, class_2 = y_pred < threshold, y_pred >= threshold
    y_pred[class_1] = 0
    y_pred[class_2] = 1
    # X_red_1, X_red_2 = X1.dot()
    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(y_pred, y):
        if label == 1 and pred == 1:
            TP += 1
        elif label == 1 and pred == 0:
            FN += 1
        elif label == 0 and pred == 1:
            FP += 1
        elif label == 0 and pred == 0:
            TN += 1
    print("Accuracy Percentage %:", ((TP + TN)/(TP+FN+FP+TN))*100)
    return X.dot(w)


df = pd.read_csv('data_banknote_authentication_a2_1.txt', sep=',', header=None)
data = np.array(df)
data = standard_scalar(data)  # Standardized Data with min=0, max=1, mean=0, var=1
np.random.shuffle(data)
d_train, d_test = np.split(data, [int(0.8 * len(data))])
X_train, y_train = d_train[:, 0:4], d_train[:, 4]
lr, epochs = np.float_power(10, -4), 8000

# Change for beta, normal etc..
w = np.random.uniform(-1/np.sqrt(X_train.shape[1]), 1/np.sqrt(X_train.shape[1]), (X_train.shape[1],))

for i in range(epochs):
    y = X_train.dot(w)
    w = w - lr*(y - y_train).T.dot(X_train)
    scores(X_train, y_train, w)

