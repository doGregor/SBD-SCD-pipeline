from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import itertools

CONFIG = {
    "cache": "./cache_dir",
    "sc_model": "./bert_sc_model",
}


class SCDetector():
    def __init__(self):
        self.label_list = ["NSC", "SC", "O"]
        self.model = AutoModelForTokenClassification.from_pretrained(pretrained_model_name_or_path=CONFIG["sc_model"],
                                                                     cache_dir=CONFIG["cache"])
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=CONFIG["sc_model"],
                                                       cache_dir=CONFIG["cache"])

    def align_sentences_with_sc_tags(self, labeled_output_tuples):
        sentence_list = []
        sentence = []
        sc_label = True
        for idx, prediction in enumerate(labeled_output_tuples):
            sentence.append(prediction[0])
            if prediction[0] == '.':
                if idx < len(labeled_output_tuples)-1:
                    sentence_list.append((sentence, sc_label))
                    sentence = []
                    if labeled_output_tuples[idx+1][1] == "SC":
                        sc_label = True
                    else:
                        sc_label = False
                else:
                    sentence_list.append((sentence, sc_label))
        return sentence_list

    def predict_speakers(self, sentence_list):
        print("[INFO] Predicting and aligning speaker change information")
        sentence_list = list(itertools.chain.from_iterable(sentence_list))
        inputs = self.tokenizer.encode(sentence_list, return_tensors="pt")
        outputs = self.model(inputs)[0]
        predictions = torch.argmax(outputs, dim=2)

        labeled_output = []
        for token, prediction in zip(sentence_list, predictions[0].tolist()[1:-1]):
            labeled_output.append((token, self.label_list[prediction]))

        return self.align_sentences_with_sc_tags(labeled_output)
