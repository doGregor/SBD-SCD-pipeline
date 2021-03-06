"""
This module contains functionality to detect sentence boundaries from unpunctuated text.
"""

from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

CONFIG = {
    "cache": "./cache_dir",
    "sbd_model": "./bert_sbd_model",
}


class SBDetector:
    def __init__(self):
        self.label_list = ["BOS", "0"]
        self.model = AutoModelForTokenClassification.from_pretrained(pretrained_model_name_or_path=CONFIG["sbd_model"],
                                                                     cache_dir=CONFIG["cache"])
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=CONFIG["sbd_model"],
                                                       cache_dir=CONFIG["cache"])

    def align_sentences(self, labeled_output_tuples):
        """
        Aligns predicted sentence boundaries with input tokens.
        :param labeled_output_tuples: list of POS-tagged tuples
        :return: list of lists where each list contains tokens of a single sentence
                 with dot for detected sentence boundaries
        """
        identified_sentences = []
        single_sentence = []
        for idx, prediction in enumerate(labeled_output_tuples):
            if prediction[1] == "BOS" and idx > 0:
                single_sentence.append('.')
                identified_sentences.append(single_sentence)
                single_sentence = []
                single_sentence.append(prediction[0])
            else:
                single_sentence.append(prediction[0])
            if idx == len(labeled_output_tuples)-1:
                single_sentence.append('.')
                identified_sentences.append(single_sentence)
        return identified_sentences

    def predict_sentence_boundaries(self, tokens):
        """
        Performs POS-tagging by using BERT SBD-model to predict sentence boundaries.
        :param tokens: list of tokens
        :return: list of lists where each list contains tokens of a single sentence
                 with dot for detected sentence boundaries
        """
        print("[INFO] Predicting and aligning sentence boundaries")
        inputs = self.tokenizer.encode(tokens, return_tensors="pt")
        outputs = self.model(inputs)[0]
        predictions = torch.argmax(outputs, dim=2)

        labeled_output = []
        for token, prediction in zip(tokens, predictions[0].tolist()[1:-1]):
            labeled_output.append((token, self.label_list[prediction]))

        return self.align_sentences(labeled_output)
