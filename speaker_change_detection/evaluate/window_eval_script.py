import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

path_to_ground_truth_logging_file = "../output/sw_pos_sc_test.txt"
path_to_predicted_logging = "test_predictions_sw.txt"


true_df_tags = []
true_df = pd.read_csv(path_to_ground_truth_logging_file,
                        delimiter=" ",
                        header=None,
                        skip_blank_lines=False)
df_list = np.split(true_df, true_df[true_df.isnull().all(1)].index)

window_df_tags = []
window_df = pd.read_csv(path_to_predicted_logging,
                        delimiter=" ",
                        header=None,
                        skip_blank_lines=False)
df_list_window = np.split(window_df, window_df[window_df.isnull().all(1)].index)

print(len(df_list), len(df_list_window))

for idx, df_single in enumerate(df_list):
    true_tag_col = df_single[1].to_numpy()
    #true_token_col = df_single[0].to_numpy()

    pred_df = df_list_window[idx]
    pred_tag_col = pred_df[1].to_numpy()
    #pred_token_col = pred_df[0].to_numpy()

    if idx == 0:
        indices = np.where(true_tag_col != "O")[0][:4]
        [true_df_tags.append(i) for i in true_tag_col[indices]]

        [window_df_tags.append(i) for i in pred_tag_col[indices]]

    elif idx == len(df_list) - 2:
        indices = np.where(true_tag_col != "O")[0][4:]
        [true_df_tags.append(i) for i in true_tag_col[indices]]

        [window_df_tags.append(i) for i in pred_tag_col[indices]]

    elif idx == len(df_list) - 1:
        pass
    else:
        index = np.where(true_tag_col != "O")[0][4]
        true_df_tags.append(true_tag_col[index])

        window_df_tags.append(pred_tag_col[index])

print(len(true_df_tags), len(window_df_tags))

true_binary = []
window_binary = []

for value in true_df_tags:
    if value == "SC":
        true_binary.append(1)
    else:
        true_binary.append(0)
for value in window_df_tags:
    if value == "SC":
        window_binary.append(1)
    else:
        window_binary.append(0)

print(len(true_binary), len(window_binary))

print("F1-score:", f1_score(true_binary, window_binary))
print("Accuracy:", accuracy_score(true_binary, window_binary))
print(classification_report(true_binary, window_binary, digits=5))
