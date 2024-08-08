from transformers import XLMRobertaForSequenceClassification, XLMRobertaConfig
import torch
from torch import nn


class IdiomaticityClassifier(nn.Module):

    def __init__(self, model, freeze=None):
        super(IdiomaticityClassifier, self).__init__()
        self.model = model
        self.freeze = freeze

        # Freeze layers if required
        if freeze is not None:
            print(f"The following layers will be frozen during fine-tuning: {', '.join(self.freeze)}.")
            self.freeze_layers(freeze)

    def freeze_layers(self, layers):
        for layer in layers:
            for param in self.model.roberta.encoder.layer[layer].parameters():
                param.requires_grad = False
        # Print to verify which layers are frozen
        for i, layer in enumerate(self.model.roberta.encoder.layer):
            is_frozen = not any(param.requires_grad for param in layer.parameters())
            print(f"Layer {i} is {'frozen' if is_frozen else 'trainable'}")

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs