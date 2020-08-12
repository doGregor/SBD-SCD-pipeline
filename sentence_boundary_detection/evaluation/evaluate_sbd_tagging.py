from sklearn.metrics import classification_report
import pandas as pd
from collections import Counter

true_df = pd.read_csv("./data/true_test.txt", sep=" ", header=None)
true_df.columns = ["token", "pos"]
true_df.dropna()
pred_df = pd.read_csv("./data/predicted_test.txt", sep=" ", header=None)
pred_df.columns = ["token", "pos"]
pred_df.dropna()

true_col = list(true_df["pos"])
pred_col = list(pred_df["pos"])

true_pos_bos = 0
true_pos_o = 0
false_pos_bos = 0
false_pos_o = 0

print("Length true: ", len(true_col))
print("Length predicted: ", len(pred_col))

for index, tag in enumerate(true_col):
    if tag is not "O" and pred_col[index] is not "O":
        true_pos_bos += 1
    if tag is "O" and pred_col[index] is "O":
        true_pos_o += 1
    if tag is not "O" and pred_col[index] is "O":
        false_pos_o += 1
    if tag is "O" and pred_col[index] is not "O":
        false_pos_bos += 1

print(classification_report(true_col, pred_col, digits=5))

print("True positive 'O': ", true_pos_o)
print("True positive 'BOS': ", true_pos_bos)
print("False positive 'O': ", false_pos_o)
print("False positive 'BOS': ", false_pos_bos)

size = len(true_col)
idx_list = [idx for idx, val in enumerate(true_col) if val is not "O"]
pred_splits = [pred_col[i: j] for i, j in zip([0] + idx_list, idx_list + ([size] if idx_list[-1] != size else []))]
true_splits = [true_col[i: j] for i, j in zip([0] + idx_list, idx_list + ([size] if idx_list[-1] != size else []))]

print("Number of sentences: ", len(true_splits))

true_pred_sentences = 0
for sentence in pred_splits:
    if Counter(sentence)["BOS"] is 1 and sentence[0] is not "O":
        true_pred_sentences += 1

print("Number of 1:1 correctly predicted sentences: ", true_pred_sentences)
