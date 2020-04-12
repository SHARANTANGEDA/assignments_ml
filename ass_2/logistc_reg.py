import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def standard_scalar(data):
    data[:, 0:4] -= np.mean(data[:, 0:4], axis=0).reshape((1, 4))
    # std = np.std(data[:, 0:4], axis=0).reshape((1, 4))
    data[:, 0:4] /= np.std(data[:, 0:4], axis=0).reshape((1, 4))
    return data


def scores(y, y_pred):
    # y_pred = 1 / (1 + np.exp(-X.dot(w)))  # sigmoid
    # class_1, class_2 = y_pred < threshold, y_pred >= threshold
    # y_pred[class_1] = 0
    # y_pred[class_2] = 1
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
    precision_score = TP / (TP + FP)
    recall = TP / (TP + FN)
    print(TP, TN, FP, FN)
    print("Precision out of 1: ", precision_score)
    print("Recall out of 1:", recall)
    print("F-Score:", (2 * precision_score * recall) / (precision_score + recall))
    print("Accuracy Percentage %:", ((TP + TN) / (TP + FN + FP + TN)) * 100)


df = pd.read_csv('data_banknote_authentication_a2_1.txt', sep=',', header=None)
data = np.array(df)
data = standard_scalar(data)  # Standardized Data with min=0, max=1, mean=0, var=1
np.random.shuffle(data)
d_train, d_test = np.split(data, [int(0.8 * len(data))])
X_train, y_train = d_train[:, 0:4], d_train[:, 4]
const = np.ones(X_train.shape[0]).reshape((X_train.shape[0], 1))
X_train = np.hstack((X_train, const))
print(X_train.shape, y_train.shape)
lr, epochs = np.float_power(10, -4), 5000

# Change for beta, normal etc..
# w = np.random.uniform(-1/np.sqrt(X_train.shape[1]), 1/np.sqrt(X_train.shape[1]), (X_train.shape[1],))  # 96.6
# w = np.random.uniform(-1/X_train.shape[1], 1/X_train.shape[1], (X_train.shape[1],))  # 97.17
# w = np.zeros(X_train.shape[1])  # 96.08
w = np.random.normal(0, 1, X_train.shape[1])  # 97.26
# w = np.random.random(X_train.shape[1])  # 96.17

for i in range(epochs):
    y = 1 / (1 + np.exp(-X_train.dot(w)))  # sigmoid
    y[y < 0.5] = 0
    y[y >= 0.5] = 1
    w -= lr * (y - y_train).T.dot(X_train)
    # scores(y_train, y)
print(w)
X_test, y_test = d_test[:, 0:4], d_test[:, 4]
const = np.ones(X_test.shape[0]).reshape((X_test.shape[0], 1))
X_test = np.hstack((X_test, const))
y = 1 / (1 + np.exp(-X_test.dot(w)))  # sigmoid
y[y < 0.5] = 0
y[y >= 0.5] = 1
scores(y_test, y)
