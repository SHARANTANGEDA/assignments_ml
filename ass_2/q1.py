import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def get_attribute_info(w):
    w = np.abs(w)
    max_ind = np.argmax(w)
    min_ind = np.argmin(w)
    attributes = {1: 'variance of Wavelet Transformed image',
                  2: 'skewness of Wavelet Transformed image',
                  3: 'curtosis of Wavelet Transformed image',
                  4: 'entropy of image '}
    print('Most Important Feature is:', attributes[int(max_ind)], 'with Col No:', max_ind, 'and weight:', w[int(max_ind)])
    print('Least Important Feature is:', attributes[int(min_ind)], 'with Col No:', min_ind, 'and weight:', w[int(min_ind)])
    

def init_params_options(X):
    # w = np.zeros(X_train.shape[1])
    # w = np.random.random(X_train.shape[1])
    # w = np.random.normal(0, 1, X_train.shape[1])
    # w = np.random.uniform(-1/X_train.shape[1], 1/X_train.shape[1], (X_train.shape[1],))
    # w = np.random.uniform(-1/np.sqrt(X_train.shape[1]), 1/np.sqrt(X_train.shape[1]), (X_train.shape[1],))
    return [np.zeros(X_train.shape[1]), np.random.random(X_train.shape[1]), np.random.normal(0, 1, X_train.shape[1]),
            np.random.uniform(-1 / X_train.shape[1], 1 / X_train.shape[1], (X_train.shape[1],)),
            np.random.uniform(-1 / np.sqrt(X_train.shape[1]), 1 / np.sqrt(X_train.shape[1]), (X_train.shape[1],))]


def perform_train(X_train, w, epochs, regularization='none'):
    for i in range(epochs):
        y = 1 / (1 + np.exp(-X_train.dot(w)))  # sigmoid
        y[y < 0.5] = 0
        y[y >= 0.5] = 1
        vector_gradient = lr * (y - y_train).T.dot(X_train)
        if regularization == 'ridge':
            vector_gradient += lr * k * w
        elif regularization == 'lasso':
            vector_gradient += lr * k * np.sign(w)
        w -= vector_gradient
    return w


def train_and_validate(X_train, w, epochs, d_valid, regularization='none'):
    if regularization == 'none':
        return perform_train(X_train, w, epochs, regularization)
    else:
        k = 0
        X_val, y_val = d_valid[:, 0:4], d_valid[:, 4]
        const = np.ones(X_val.shape[0]).reshape((X_val.shape[0], 1))
        X_val = np.hstack((X_val, const))
        accuracy, best_k, best_w = 0, 0, w
        while k < 1:
            w_curr = perform_train(X_train, w, epochs, regularization)
            y = 1 / (1 + np.exp(-X_val.dot(w_curr)))  # sigmoid
            y[y < 0.5] = 0
            y[y >= 0.5] = 1
            acc = scores(y_val, y)[0]
            if accuracy < acc:
                accuracy = acc
                best_k = k
                best_w = w_curr
            k += 0.1
        return best_w, best_k, accuracy


def standard_scalar(data):
    data[:, 0:4] -= np.mean(data[:, 0:4], axis=0).reshape((1, 4))
    # std = np.std(data[:, 0:4], axis=0).reshape((1, 4))
    data[:, 0:4] /= np.std(data[:, 0:4], axis=0).reshape((1, 4))
    return data


def scores(y, y_pred):
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
    accuracy = ((TP + TN) / (TP + FN + FP + TN)) * 100
    f_score = (2 * precision_score * recall) / (precision_score + recall)
    # print(TP, TN, FP, FN)
    # print("F-Score:", )
    # print("Accuracy Percentage %:", accuracy)
    return accuracy, f_score, TP, TN, FP, FN


df = pd.read_csv('data_banknote_authentication_a2_1.txt', sep=',', header=None)
data = np.array(df)
data = standard_scalar(data)  # Standardized Data with min=0, max=1, mean=0, var=1
np.random.shuffle(data)
d_train, d_val, d_test = np.split(data, [int(0.7 * len(data)), int(0.8 * len(data))])
X_train, y_train = d_train[:, 0:4], d_train[:, 4]
const = np.ones(X_train.shape[0]).reshape((X_train.shape[0], 1))
X_train = np.hstack((X_train, const))
lr, epochs = np.float_power(10, -4), 3000
k = 0
# Change for beta, normal etc..
# w = np.random.uniform(-1/np.sqrt(X_train.shape[1]), 1/np.sqrt(X_train.shape[1]), (X_train.shape[1],))
# w = np.random.uniform(-1/X_train.shape[1], 1/X_train.shape[1], (X_train.shape[1],))
# w = np.zeros(X_train.shape[1])
w_init = np.random.normal(0, 1, X_train.shape[1])
# w = np.random.random(X_train.shape[1])

# WITHOUT REGULARIZATION
w = train_and_validate(X_train, w_init, epochs, d_val)
X_test, y_test = d_test[:, 0:4], d_test[:, 4]
const = np.ones(X_test.shape[0]).reshape((X_test.shape[0], 1))
X_n_test = np.hstack((X_test, const))
y = 1 / (1 + np.exp(-X_n_test.dot(w)))  # sigmoid
y[y < 0.5] = 0
y[y >= 0.5] = 1
accuracy, f_score, TP, TN, FP, FN = scores(y_test, y)
print("*************** Without regularization ***************")
print('Test Accuracy: ', accuracy)
print('Test F_Score: ', f_score)
get_attribute_info(w)
print('TP, TN, FP, FN in same order: ', TP, TN, FP, FN)
print("******************************************************")

# RIDGE Regularization
w, best_k, accuracy_val = train_and_validate(X_train, w_init, epochs, d_val, 'ridge')
X_test, y_test = d_test[:, 0:4], d_test[:, 4]
const = np.ones(X_test.shape[0]).reshape((X_test.shape[0], 1))
X_n_test = np.hstack((X_test, const))
y = 1 / (1 + np.exp(-X_n_test.dot(w)))  # sigmoid
y[y < 0.5] = 0
y[y >= 0.5] = 1
accuracy, f_score, TP, TN, FP, FN = scores(y_test, y)
print("*************** Ridge regularization *****************")
print('Best Lambda for maximum accuracy: ', best_k)
print('Best Validation Accuracy: ', accuracy_val)
print('Test Accuracy: ', accuracy)
print('Test F_Score: ', f_score)
get_attribute_info(w)
print('TP, TN, FP, FN in same order: ', TP, TN, FP, FN)
print("******************************************************")

# Lasso Regularization
w, best_k, accuracy_val = train_and_validate(X_train, w_init, epochs, d_val, 'lasso')
X_test, y_test = d_test[:, 0:4], d_test[:, 4]
const = np.ones(X_test.shape[0]).reshape((X_test.shape[0], 1))
X_n_test = np.hstack((X_test, const))
y = 1 / (1 + np.exp(-X_n_test.dot(w)))  # sigmoid
y[y < 0.5] = 0
y[y >= 0.5] = 1
accuracy, f_score, TP, TN, FP, FN = scores(y_test, y)
print("*************** Lasso regularization *****************")
print('Best Lambda for maximum accuracy: ', best_k)
print('Best Validation Accuracy: ', accuracy_val)
print('Test Accuracy: ', accuracy)
print('Test F_Score: ', f_score)
get_attribute_info(w)
print('TP, TN, FP, FN in same order: ', TP, TN, FP, FN)
print("******************************************************")
