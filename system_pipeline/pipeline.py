"""
Full pipeline module that puts together functionality to preprocess inputs,
predict sentence boundaries and speaker changes and save it as .txt file in
output_dir.
"""

import os
from text_processing_functionality import get_tokenized_data, get_predictions_as_sentences, save_predictions
from sentence_boundary_detection import SBDetector
from speaker_change_detection import SCDetector


CONFIG = {
    "input": "./input_dir",
    "output": "./output_dir",
    "working": "./working_dir",
    "sequence_length": 64,

    "cache": "./cache_dir",
    "sbd_model": "./bert_sbd_model"
}


if __name__ == '__main__':

    sb_detector = SBDetector()
    sc_detector = SCDetector()

    for file in os.listdir(CONFIG['input']):
        if file.endswith(".txt"):
            print("[INFO] Processing following file:", file)
            path_of_current_file = CONFIG['input'] + "/" + str(file)

            tokens = get_tokenized_data(path_of_current_file)
            predicted_sentences = sb_detector.predict_sentence_boundaries(tokens=tokens)
            tokens_with_sc_labels = sc_detector.predict_speakers(predicted_sentences)

            output = get_predictions_as_sentences(tokens_with_sc_labels)
            output_file_name = "processed_" + file
            save_predictions(output, output_file_name)
            #print(output)
        else:
            print("[WARNING] Skipping following file since it isn't in .txt format:", file)
