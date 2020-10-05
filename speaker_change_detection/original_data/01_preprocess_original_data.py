import os

# specify location of folders that are created by data preprocessing of Meng et al.
# run this process for each folder: train test val and specify output file name
directory = "../DATA/val"
output_name = "org_sc_val.txt"


speaker_change_label = "SC "
train_file = open(output_name, "w")
for file in os.listdir(directory):
    with open(directory+"/"+file) as curr_file:
        curr_label = None
        for idx, line in enumerate(curr_file):
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
