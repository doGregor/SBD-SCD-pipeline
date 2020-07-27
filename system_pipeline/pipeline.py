import os
from text_processing_functionality import get_tokenized_data
from sentence_boundary_detection import SBDDetector


CONFIG = {
    "input": "./input_dir",
    "output": "./output_dir",
    "working": "./working_dir",
    "sequence_length": 64,

    "cache": "./cache_dir",
    "sbd_model": "./bert_sbd_model"
}


if __name__ == '__main__':

    sbd_detector = SBDDetector()

    for file in os.listdir(CONFIG['input']):
        print("Processing following file:", file)
        path_of_current_file = CONFIG['input'] + "/" + str(file)

        tokens = get_tokenized_data(path_of_current_file)

        predicted_list = sbd_detector.predict_sentence_boundaries(tokens=tokens)

        print(predicted_list)

# TODO: remove ' from input text and refactor output
