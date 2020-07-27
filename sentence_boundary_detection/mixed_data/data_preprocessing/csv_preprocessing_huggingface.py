import pandas as pd
import os
import numpy as np


def save_txt(data, filename, path="./output"):
    path = os.path.join(path, filename + ".txt")
    data.to_csv(path_or_buf=path, sep=" ", index=False, header=False)
    f = open(path, "r+")
    file_data = f.read()
    file_data = file_data.splitlines()
    f.truncate(0)
    for line in file_data:
        line = line.rstrip()
        if not line:
            f.write("\n")
        else:
            f.write(line+"\n")
    f.close()


def prepare_sequences(data_frame, sequence_length):
    df = data_frame.dropna()
    sentences = []
    for g, df_p in df.groupby(np.arange(df.shape[0]) // sequence_length):
        if len(df_p["pos"].tolist()) is sequence_length:
            df_p = df_p.append(pd.Series(), ignore_index=True)
            sentences.append(df_p)
    return pd.concat(sentences)


train = pd.read_csv("./output/train.csv")
train_sequences = prepare_sequences(train, 64)
save_txt(train_sequences, "train_final")

dev = pd.read_csv("./output/dev.csv")
dev_sequences = prepare_sequences(dev, 64)
save_txt(dev_sequences, "dev_final")

test = pd.read_csv("./output/test.csv")
test_sequences = prepare_sequences(test, 64)
save_txt(test_sequences, "test_final")
