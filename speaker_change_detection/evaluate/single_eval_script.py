from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import numpy as np

true_df = pd.read_csv("../output/ground_truth_no_sw.txt", sep=" ", header=None)
true_df.columns = ["token", "pos"]
true_df.dropna()
pred_df = pd.read_csv("test_predictions_no_sw.txt", sep=" ", header=None)
pred_df.columns = ["token", "pos"]
pred_df.dropna()

true_col = list(true_df["pos"])
pred_col = list(pred_df["pos"])

print("consistency", len(true_col), len(pred_col))

true_log = []
pred_log = []

for idx, pos_tag in enumerate(true_col):
    if pos_tag == "SC":
        true_log.append(1)
        if pred_col[idx] == "SC":
            pred_log.append(1)
        else:
            pred_log.append(0)
    # for evaluation modified files with NSC tags that indicate sentence beginning are required
    elif pos_tag == "NSC":
        true_log.append(0)
        if pred_col[idx] == "SC":
            pred_log.append(1)
        else:
            pred_log.append(0)

print("consistency", len(true_log), len(pred_log))

true_log = np.asarray(true_log)
pred_log = np.asarray(pred_log)

correctly_detected_changes = np.intersect1d(np.where(pred_log == 1), np.where(true_log == 1)).size
detected_changes = np.where(pred_log == 1)[0].size
all_changes = np.intersect1d(np.where(true_log == 1), np.where(pred_log == 0)).size + correctly_detected_changes

P = correctly_detected_changes/detected_changes
R = correctly_detected_changes/all_changes

F1 = 2*P * R/(P+R)

print("F1-score:", F1)
print("Accuracy:", accuracy_score(true_log, pred_log))
print(classification_report(true_log, pred_log, digits=5))
