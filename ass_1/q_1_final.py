import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))


def plot_gaussian(X0, X1, intersection):
    m0, m1 = np.mean(X0), np.mean(X1)
    std0, std1 = np.sqrt(np.mean((X0 - m0) ** 2)), np.sqrt(np.mean((X1 - m1) ** 2))
    interval = np.linspace(min(m0 - 3 * std0, m1 - 3 * std1),
                           max(m0 + 3 * std0, m1 + 3 * std1), 1000)
    ax1.set_xlim(min(m0 - 3 * std0, m1 - 3 * std1), max(m0 + 3 * std0, m1 + 3 * std1))
    ax1.grid()
    ax1.set_title('Normal Distribution of Each Class\n$\\regular_{Green \ dot\ shows\ the\ intersection\ of\ curves}$',
                  fontsize=20)
    ax1.scatter(intersection, norm.pdf(intersection, m0, std0), c="green")
    ax1.plot(interval, norm.pdf(interval, m0, std0), 'k', linewidth=1, color='red')
    ax1.grid()
    ax1.plot(interval, norm.pdf(interval, m1, std1), 'k', linewidth=1, color='blue')


def scores(X, y, w, threshold):
    y_pred = X.dot(w)
    class_1, class_2 = y_pred < threshold, y_pred >= threshold
    y_pred[class_1] = 0
    y_pred[class_2] = 1
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
    precision_score = TP / (TP + FP)
    recall = TP / (TP + FN)
    print("Precision out of 1: ", precision_score)
    print("Recall out of 1:", recall)
    print("F-Score:", (2 * precision_score * recall) / (precision_score + recall))
    print("Accuracy Percentage %:", ((TP + TN) / (TP + FN + FP + TN)) * 100)
    return X.dot(w)


# This will yield a quadratic equation of form ax**2+bx+c, so we find a, b, c and their roots here
def solve_gaussian_equations(mean_1, std_1, mean_2, std_2):
    a = (1 / (2 * std_1 ** 2)) - (1 / (2 * std_2 ** 2))
    b = (mean_2 / (std_2 ** 2)) - (mean_1 / (std_1 ** 2))
    c = ((mean_1 ** 2) / (2 * std_1 ** 2)) - \
        ((mean_2 ** 2) / (2 * std_2 ** 2)) - np.log(std_2 / std_1)
    return np.roots([a, b, c])


data_path = str(
    input("Enter relative path to the dataset a1_d1.csv, a1_d2.csv etc. to make prediction:\n"))
df = 0
try:
    df = pd.read_csv(data_path, sep=',', header=None)
except:
    print("Please give the correct path, there was an error !!!")
    exit(0)
data = np.array(df)
cols = len(df.columns)
X = df[df.columns[0:cols - 1]]
y = np.array(df[df.columns[cols - 1]]).reshape(-1, 1)
d_0, d_1 = data[data[:, cols - 1] == 0], data[data[:, cols - 1] == 1]
X_0 = d_0[:, 0:cols - 1]
X_1 = d_1[:, 0:cols - 1]
mean0, mean1 = np.mean(X_0, axis=0).reshape(
    (1, cols - 1)), np.mean(X_1, axis=0).reshape((1, cols - 1))
S_w = (X_0 - mean0).T.dot((X_0 - mean0)) + (X_1 - mean1).T.dot((X_1 - mean1))
w = np.linalg.inv(S_w).dot((mean1 - mean0).T)  # Parameter w
X_0_trans, x_1_trans = X_0.dot(w), X_1.dot(w)
X0_new_mean = np.mean(X_0_trans)
X0_new_std = np.sqrt(np.mean((X_0_trans - X0_new_mean) ** 2))
X1_new_mean = np.mean(x_1_trans)
X1_new_std = np.sqrt(np.mean((x_1_trans - X1_new_mean) ** 2))
intersection = solve_gaussian_equations(X0_new_mean, X0_new_std, X1_new_mean, X1_new_std)
threshold = intersection[1]  # Threshold
print("Threshold:", threshold)
X, y = data[:, 0:cols - 1], data[:, [cols - 1]]
X_reduced = scores(X, y, w, threshold)
plot_gaussian(X_0_trans, x_1_trans, intersection)
y_zeroes = np.zeros(X_reduced.shape[0])
# ax2.scatter(X_reduced, y_zeroes, marker='.')
plt.scatter(X_0_trans, [0] * len(X_0_trans), label="dots", color="red", marker=".", s=3)
plt.scatter(x_1_trans, [0] * len(x_1_trans), label="dots", color="blue", marker=".", s=3)
ax2.set_title(
    'Fisher Discriminant Analysis\n$\\regular_{Test \ Points\ on\ new\ Axis}$', fontsize=20)
plt.show()
