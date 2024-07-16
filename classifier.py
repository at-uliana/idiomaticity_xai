from transformers import XLMRobertaForSequenceClassification
from torch import nn, optim


class IdiomaticityClassifier(nn.Module):

    def __init__(self, configs):
        super(IdiomaticityClassifier, self).__init__()
        self.transformer = XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-base',
                                                                               num_labels=2,
                                                                               output_hidden_states=False,
                                                                               output_attentions=False
                                                                               )
        self.batch_size = configs.batch_size

        # Fine-tune the model or not while learning the classifier
        if configs.freeze:
            self.freeze_pretrained()
        else:
            self.unfreeze_pretrained()

    def freeze_pretrained(self):
        # doesn't work
        for param in self.transformer.named_parameters():
            param[1].requires_grad = False

    def unfreeze_pretrained(self):
        # doesn't work
        for param in self.transformer.named_parameters():
            param[1].requires_grad = True

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs
