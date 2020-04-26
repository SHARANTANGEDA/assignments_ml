import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))


def plot_gaussian(X0, X1, intersection):
    m0, m1 = np.mean(X0), np.mean(X1)
    std0, std1 = np.sqrt(np.mean((X0 - m0) ** 2)), np.sqrt(np.mean((X1 - m1) ** 2))
    interval = np.linspace(min(m0 - 3*std0, m1 - 3*std1), max(m0+3*std0, m1 + 3*std1), 1000)
    ax1.set_xlim(min(m0 - 3*std0, m1 - 3*std1), max(m0+3*std0, m1 + 3*std1))
    ax1.grid()
    ax1.set_title('Normal Distribution of Each Class\n$\\regular_{Green \ dot\ shows\ the\ intersection\ of\ curves}$',
                  fontsize=20)
    ax1.scatter(intersection, norm.pdf(intersection, m0, std0), c="green")
    ax1.plot(interval, norm.pdf(interval, m0, std0), 'k', linewidth=1, color='red')
    ax1.grid()
    ax1.plot(interval, norm.pdf(interval, m1, std1), 'k', linewidth=1, color='blue')


def calc_s_w_terms(X, mean):
    print("SUM:", (X - mean).T.dot((X - mean)))

    return (X - mean).T.dot((X - mean))


def scores(X, y, w, threshold):
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
    print(TP, TN, FP, FN)
    precision_score = TP/(TP + FP)
    recall = TP/(TP + FN)
    print("Precision out of 1: ", precision_score)
    print("Recall out of 1:", recall)
    print("F-Score:", (2*precision_score*recall)/(precision_score + recall))
    print("Accuracy Percentage %:", ((TP + TN)/(TP+FN+FP+TN))*100)
    return X.dot(w)


# This will yield a quadratic equation of form ax**2+bx+c, so we find a, b, c and their roots here
def solve_gaussian_equations(mean_1, std_1, mean_2, std_2):
    a = (1 / (2 * std_1 ** 2)) - (1 / (2 * std_2 ** 2))
    b = (mean_2 / (std_2 ** 2)) - (mean_1 / (std_1 ** 2))
    c = ((mean_1 ** 2) / (2 * std_1 ** 2)) - \
        ((mean_2 ** 2) / (2 * std_2 ** 2)) - np.log(std_2/std_1)
    return np.roots([a, b, c])


df = pd.read_csv('./a1_data/a1_d1.csv', sep=',', header=None)
data = np.array(df)
X = df[df.columns[0:2]]
y = np.array(df[df.columns[2]]).reshape(-1, 1)
d_train, d_test = np.split(data, [int(0.8 * len(data))])
d_train0, d_train1 = d_train[d_train[:, 2] == 0], d_train[d_train[:, 2] == 1]
X_train0 = d_train0[:, [0, 1]]
X_train1 = d_train1[:, [0, 1]]
mean0, mean1 = np.mean(X_train0, axis=0).reshape((1, 2)), np.mean(X_train1, axis=0).reshape((1, 2))
S_w = calc_s_w_terms(X_train0, mean0) + calc_s_w_terms(X_train1, mean1)
w = np.linalg.inv(S_w).dot((mean1 - mean0).T)  # Parameter w
print(w)
X_train0_trans, X_train1_trans = X_train0.dot(w), X_train1.dot(w)
X0_new_mean = np.mean(X_train0_trans)
X0_new_std = np.sqrt(np.mean((X_train0_trans - X0_new_mean) ** 2))
X1_new_mean = np.mean(X_train1_trans)
X1_new_std = np.sqrt(np.mean((X_train1_trans - X1_new_mean) ** 2))
print("Res", X0_new_mean, X0_new_std, X1_new_mean, X1_new_std)
intersection = solve_gaussian_equations(X0_new_mean, X0_new_std, X1_new_mean, X1_new_std)
print("POINT:", intersection)
threshold = intersection[1]  # Threshold
X_test, y_test = d_test[:, [0, 1]], d_test[:, [2]]
X_reduced = scores(X_test, y_test, w, threshold)
# plot_data_sample()
plot_gaussian(X_train0_trans, X_train1_trans, intersection)
y_zeroes = np.zeros(X_reduced.shape[0])
# ax2.scatter(X_reduced, y_zeroes, marker='.')
plt.scatter(X_train0_trans, [0] * len(X_train0_trans), label="dots", color="red", marker=".", s=3)
plt.scatter(X_train1_trans, [0] * len(X_train1_trans), label="dots", color="blue", marker=".", s=3)
ax2.set_title(
    'Fisher Discriminant Analysis\n$\\regular_{Test \ Points\ on\ new\ Axis}$', fontsize=20)
plt.show()
