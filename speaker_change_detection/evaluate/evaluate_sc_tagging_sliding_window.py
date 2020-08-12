from sklearn.metrics import classification_report, accuracy_score
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import pandas as pd
import numpy as np

file_path = "CoNLL_formated_data.txt"

CONFIG = {
    "sbd_model": "../original_sc_bert"
}

label_list = ["NSC", "SC", "O"]

model = AutoModelForTokenClassification.from_pretrained(pretrained_model_name_or_path=CONFIG["sbd_model"])
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=CONFIG["sbd_model"])

df = pd.read_csv(file_path, sep=" ", header=None)
df.columns = ["token", "pos"]
df.dropna()

indices = df.index[df['pos'] != "O"].tolist()

df_list = np.split(df, indices)
del df_list[0]

true = []
predicted = []

for idx in range(0, len(df_list)):
    print(idx/len(df_list))
    if idx == 0:
        sequence = pd.concat(df_list[0:7])
    elif idx != 0 and idx < len(df_list) - 6:
        sequence = pd.concat(df_list[idx:idx + 7])
    elif idx > len(df_list) - 6:
        break
    else:
        sequence = pd.concat(df_list[idx:])

    tokens = list(sequence["token"])
    pos_tags = np.asarray(sequence["pos"])

    inputs = tokenizer.encode(tokens, return_tensors="pt")
    outputs = model(inputs)[0]
    predictions = torch.argmax(outputs, dim=2)
    labels = []
    for token, prediction in zip(tokens, predictions[0].tolist()[1:-1]):
        labels.append(label_list[prediction])

    if idx == 0:
        relevant_idx = np.where(pos_tags != "O")[0][:4]
        relevant_idx = list(relevant_idx)
    elif idx != 0 and idx < len(df_list) - 6:
        relevant_idx = np.where(pos_tags != "O")[0][3]
    else:
        relevant_idx = np.where(pos_tags != "O")[0][:]
        relevant_idx = list(relevant_idx)

    if type(relevant_idx) == list:
        for value in relevant_idx:
            if pos_tags[value] == "SC":
                true.append(1)
            else:
                true.append(0)
            if labels[value] == "SC":
                predicted.append(1)
            else:
                predicted.append(0)
    else:
        if pos_tags[relevant_idx] == "SC":
            true.append(1)
        else:
            true.append(0)
        if labels[relevant_idx] == "SC":
            predicted.append(1)
        else:
            predicted.append(0)


print(len(true), len(predicted))

print(classification_report(true, predicted))
print("Accuracy", accuracy_score(true, predicted))
