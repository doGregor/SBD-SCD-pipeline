import os
import re

directory = "../DATA/train"
output_name = "sc_train.txt"


speaker_change_label = "SC "
train_file = open(output_name, "w")
for file in os.listdir(directory):
    with open(directory+"/"+file) as curr_file:
        curr_label = None
        for idx, line in enumerate(curr_file):
            line = re.sub('[,`´’"#&()*+/:;<=>@^_{|}~-]', "", line)
            line = re.sub("[.]{2,}", ".", line)
            line = re.sub("[!?]", ".", line)
            line.rstrip()
            label = line[-6:]
            line = line[:-6]
            if idx == 0:
                to_append = speaker_change_label + line
                curr_label = label
            elif curr_label != label:
                curr_label = label
                to_append = speaker_change_label + line
            else:
                to_append = line

            train_file.write(to_append)
            train_file.write("\n")
        curr_file.close()

train_file.close()
