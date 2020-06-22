from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

CONFIG = {
    "cache": "./cache_dir",
    "sbd_model": "./bert_sbd_model",
}


class SBDDetector:
    label_list = ["BOS", "0"]
    model = AutoModelForTokenClassification.from_pretrained(pretrained_model_name_or_path=CONFIG["sbd_model"],
                                                            cache_dir=CONFIG["cache"])
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=CONFIG["sbd_model"],
                                              cache_dir=CONFIG["cache"])

    def __init__(self):
        pass

    def predict_sentence_boundaries(self, tokens):
        inputs = self.tokenizer.encode(tokens, return_tensors="pt")
        outputs = self.model(inputs)[0]
        predictions = torch.argmax(outputs, dim=2)

        labeled_output = []
        for token, prediction in zip(tokens, predictions[0].tolist()[1:-1]):
            labeled_output.append((token, self.label_list[prediction]))

        return labeled_output
