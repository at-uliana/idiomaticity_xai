import transformers
from transformers import get_linear_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup, \
    XLMRobertaForSequenceClassification
import torch
from torch import nn, optim


class IdiomaticityClassifier(nn.Module):
    '''Network definition '''

    def __init__(self, args):
        super(IdiomaticityClassifier, self).__init__()
        self.transformer = XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-base',
                                                                               num_labels=2,
                                                                               output_hidden_states=False,
                                                                               output_attentions=False
                                                                               )
        self.batch_size = args.batch_size

        # Fine-tune the model or not while learning the classifier
        if args.freeze_pretrained:
            self.freeze_pretrained()
        else:
            self.unfreeze_pretrained()

    def freeze_pretrained(self):
        for param in self.transformer.named_parameters():
            param[1].requires_grad = False

    def unfreeze_pretrained(self):
        for param in self.transformer.named_parameters():
            param[1].requires_grad = True

    # def forward(self, input_ids, attention_mask, label):
    def forward(self, input_ids, attention_mask, labels):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs
