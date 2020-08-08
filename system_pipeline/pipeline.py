import os
from text_processing_functionality import get_tokenized_data, get_predictions_as_sentences
from sentence_boundary_detection import SBDDetector
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

    sbd_detector = SBDDetector()
    sc_detector = SCDetector()

    for file in os.listdir(CONFIG['input']):
        print("[INFO] Processing following file:", file)
        path_of_current_file = CONFIG['input'] + "/" + str(file)

        tokens = get_tokenized_data(path_of_current_file)

        predicted_sentences = sbd_detector.predict_sentence_boundaries(tokens=tokens)

        tokens_with_sc_labels = sc_detector.predict_speakers(predicted_sentences)

        output = get_predictions_as_sentences(tokens_with_sc_labels)
        print(output)
