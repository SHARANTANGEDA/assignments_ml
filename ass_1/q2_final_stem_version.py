import numpy as np
import pandas as pd
from stemming.porter2 import stem


def scores(y, y_pred, fold_ind):
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
    print("Fold " + str(fold_ind) + ": TP, TN, FP, FN = " +
          str(TP) + ", " + str(TN) + ", " + str(FP) + ", " + str(FN))
    return accuracy, f_score


def sentence_to_stem(sentence):
    sentence = sentence[0]
    sentence_flat = sentence.replace('\r', '\n').replace('\n', ' ').lower()
    punctuation = (',', "'", '"', ",", ';', ':', '.', '?',
                   '!', '(', ')', '{', '}', '/', '_', '|', '-')
    for p in punctuation:
        sentence_flat = sentence_flat.replace(p, '')
    words = filter(lambda x: x.strip() != '', sentence_flat.split(' '))
    words = map(lambda x: stem(x), words)
    s = ' '.join(words)
    return s


def fill_vocab(sentence, vocab, pos_vocab, vocab_0, y, wc_total, wc_pos, laplace_constant):
    words = sentence.split(' ')
    for word in words:
        wc_total += 1
        if word not in vocab:
            vocab[word] = laplace_constant
            pos_vocab[word] = laplace_constant
            vocab_0[word] = laplace_constant
        vocab[word] += 1
        if y == 1:
            pos_vocab[word] += 1
            wc_pos += 1
        if y == 0:
            vocab_0[word] += 1
    return wc_total, wc_pos


def calc_prob(X, y, laplace_constant=1):
    prob_0, prob_1 = y[y == 0].shape[0] / y.shape[0], y[y == 1].shape[0] / y.shape[0]
    vocab, vocab_1, vocab_0, word_cnt_total, word_cnt_1 = {}, {}, {}, 0, 0
    for ind, sentence in enumerate(X):
        word_cnt_total, word_cnt_1 = fill_vocab(sentence[0], vocab, vocab_1, vocab_0, y[ind], word_cnt_total,
                                                word_cnt_1, laplace_constant)
    print(word_cnt_1, word_cnt_total, len(vocab), len(vocab_0), len(vocab_1))
    word_prob_1 = {key: value / (word_cnt_1 + len(vocab)) for key, value in vocab_1.items()}
    word_prob_0 = {key: value / (word_cnt_total - word_cnt_1 + len(vocab))
                   for key, value in vocab_0.items()}
    log_prior_0, log_prior_1 = prob_0, prob_1
    return vocab, vocab_0, vocab_1, word_prob_0, word_prob_1, word_cnt_total, word_cnt_1, log_prior_0, log_prior_1


df = pd.read_csv('a1_d3.txt', delimiter='\t', header=None)
data = np.array(df)
X, y = data[:, 0].reshape((len(data), 1)), data[:, 1]
X = np.apply_along_axis(sentence_to_stem, 1, X).reshape((len(X), 1))
print(X.shape)
cross_valid_lines = np.split(X, (int(0.2 * len(X)), int(0.4 * len(X)),
                                 int(0.6 * len(X)), int(0.8 * len(X))))
cross_valid_target = np.split(y, (int(0.2 * len(y)), int(0.4 * len(y)),
                                  int(0.6 * len(y)), int(0.8 * len(y))))
predictions = []
accuracy_list = []
F_score_list = []
for idx in range(5):
    X_test, y_test = cross_valid_lines[idx], cross_valid_target[idx]
    X_train = np.concatenate([x for i, x in enumerate(cross_valid_lines) if i != idx])
    y_train = np.concatenate([x for i, x in enumerate(cross_valid_target) if i != idx])
    vocab, vocab_0, vocab_1, word_prob_0, word_prob_1, word_cnt_total, word_cnt_1, log_prior_0, log_prior_1 = calc_prob(
        X_train, y_train)
    y_pred = []
    for ind in range(len(X_test)):
        sent = X_test[ind][0].split(' ')
        prob_sum_0, prob_sum_1 = log_prior_0, log_prior_1
        for word in sent:
            if word in vocab_0:
                prob_sum_0 *= word_prob_0[word]
            if word in vocab_1:
                prob_sum_1 *= word_prob_1[word]
        if prob_sum_0 > prob_sum_1:
            y_pred.append(0)
        else:
            y_pred.append(1)
    predictions.append(y_pred)
    accuracy, f_score = scores(y_test, y_pred, idx + 1)
    accuracy_list.append(accuracy)
    F_score_list.append(f_score)
print("accuracy in 5 folds testing = " + str(accuracy_list))
mean, std = np.mean(np.array(accuracy_list)), np.std(np.array(accuracy_list))
print("MEAN, STD:", mean, std)
print("Accuracy = " + str(round(mean, 5)) + "+-" + str(round(std, 5)))
print("F Score in 5 folds " + str(F_score_list))
mean_f, std_f = np.mean(np.array(F_score_list)), np.std(np.array(F_score_list))
print("F-score = " + str(round(mean_f, 5)) + "+-" + str(round(std_f, 5)))
