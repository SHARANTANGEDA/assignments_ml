from Sentence_Stem import sentence_to_stem

f = open("a1_d3.txt", "r")

lines = f.readlines()

stemmed_lines = []
target = []

for line in lines:
    l = line.strip().split()
    t = l[len(l) - 1]
    del (l[len(l) - 1])
    line_removing_t = " ".join(l)
    target.append(int(t))
    stem = sentence_to_stem(line_removing_t)
    stemmed_lines.append(stem)


def cal_prob(part_lines, part_target):
    word_to_freq = {}
    word_to_one = {}
    one_count = 0
    total_word_cnt = 0
    total_word_cnt_one = 0
    for i in range(len(part_lines)):
        line = part_lines[i]
        for j in range(len(line)):
            word = line[j]
            if word not in word_to_freq.keys():
                word_to_freq[word] = 0
                word_to_one[word] = 0

            word_to_freq[word] += 1
            if part_target[i] == 1:
                word_to_one[word] += 1

        if part_target[i] == 1:
            one_count += 1
            total_word_cnt_one += (len(line))
        total_word_cnt += (len(line))

    word_to_prob = {}
    for word in word_to_freq.keys():
        word_to_prob[word] = word_to_freq[word] / total_word_cnt

    word_after_one_prob = {}
    for word in word_to_one.keys():
        word_after_one_prob[word] = word_to_one[word] / total_word_cnt_one

    return word_to_prob, word_after_one_prob, one_count, total_word_cnt


parts_lines = [[], [], [], [], []]
parts_target = [[], [], [], [], []]
for i in range(len(stemmed_lines)):
    parts_lines[i % 5].append(stemmed_lines[i])
    parts_target[i % 5].append(target[i])

predictions = []
accuracy_list = []
F_score_list = []

for i in range(5):
    train_x = []
    train_y = []
    test_x = parts_lines[i]
    test_y = parts_target[i]
    for j in range(5):
        if j != i:
            train_x = train_x + parts_lines[j]
            train_y = train_y + parts_target[j]
    (word_to_prob, word_after_one_prob, one_count, total_word_cnt) = cal_prob(train_x, train_y)

    predicted_y = []
    for j in range(len(test_x)):
        line = test_x[j]
        numerator = 1
        denominator = 1
        for k in range(len(line)):
            word = line[k]
            if (word in word_to_prob.keys()):
                numerator *= (word_after_one_prob[word])
                denominator *= (word_to_prob[word])
        numerator *= (one_count / len(train_y))

        prob_1 = numerator / denominator

        if (prob_1 >= 0.5):
            predicted_y.append(1)
        else:
            predicted_y.append(0)
    predictions.append(predicted_y)

    match_cnt = 0
    for j in range(len(predicted_y)):
        if (predicted_y[j] == test_y[j]):
            match_cnt += 1
    accuracy = match_cnt / len(predicted_y)

    TP, TN, FP, FN = 0, 0, 0, 0

    for pred, label in zip(predicted_y, test_y):
        if label == 1 and pred == 1:
            TP += 1
        elif label == 1 and pred == 0:
            FN += 1
        elif label == 0 and pred == 1:
            FP += 1
        elif label == 0 and pred == 0:
            TN += 1

    print("Fold " + str(i + 1) + ": TP, TN, FP, FN = " + str(TP) +
          ", " + str(TN) + ", " + str(FP) + ", " + str(FN))
    precision_score = TP / (TP + FP)
    recall = TP / (TP + FN)
    f_score = (2 * precision_score * recall) / (precision_score + recall)
    accuracy_list.append(accuracy)
    F_score_list.append(f_score)

print("accuracy in 5 folds testing = " + str(accuracy_list))

mean = 0
for i in range(len(accuracy_list)):
    mean += accuracy_list[i]

mean /= (len(accuracy_list))

std = 0
for i in range(len(accuracy_list)):
    std += ((accuracy_list[i] - mean) ** 2)

std /= (len(accuracy_list))
std ** (0.5)

print("Accuracy = " + str(round(mean, 5)) + "+-" + str(round(std, 5)))

print("F Score in 5 folds " + str(F_score_list))

mean_f = 0
for i in range(len(F_score_list)):
    mean_f += F_score_list[i]

mean_f /= (len(F_score_list))

std_f = 0
for i in range(len(F_score_list)):
    std_f += ((F_score_list[i] - mean_f) ** 2)

std_f /= (len(F_score_list))
std_f ** (0.5)
print("F-score = " + str(round(mean_f, 5)) + "+-" + str(round(std_f, 5)))
