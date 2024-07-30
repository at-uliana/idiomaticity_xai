import torch
import json
import os
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from utils import are_all_model_parameters_on_gpu, are_all_model_buffers_on_gpu


class IdiomaticityTester:

    def __init__(self, model, device, test_loader):
        self.model = model
        self.test_loader = test_loader
        self.device = device

    def test(self):
        # Prepare data structures to store results
        predictions = []
        correct_labels = []
        total_loss = []

        # Put model in inference mode
        self.model.eval()

        with torch.no_grad():
            for batch in self.test_loader:
                input_ids, attention_mask, labels = batch

                # Send tensors to the device
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)

                # Get outputs, calculate loss and predictions
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss, logits = outputs.loss, outputs.logits
                batch_predictions = torch.argmax(torch.softmax(logits, dim=1), dim=1)
                predictions.extend(batch_predictions.tolist())
                correct_labels.extend(labels.tolist())
                total_loss.append(loss)
        test_results = {
            'predictions': predictions,
            'loss': total_loss,
            'accuracy': accuracy_score(correct_labels, predictions),
            'f1-score': f1_score(correct_labels, predictions),
            'precision': precision_score(correct_labels, predictions),
            'recall': recall_score(correct_labels, predictions)
        }
        return test_results


# def get_predictions(self, external_dataloader=None):
#     if external_dataloader:
#         test_loader = external_dataloader
#     else:
#         test_loader = self.test_loader
#
#     self.model.eval()
#     predictions = []
#     prediction_probs = []
#     true_labels = []
#
#     with torch.no_grad():
#         i = 0
#         for batch in test_loader:
#             input_ids, attention_mask, labels = batch
#             outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
#             loss, logits = outputs.loss, outputs.logits
#
#             # Save `true` labels
#             true_labels.extend(labels)
#
#             # Get probabilities
#             probs = torch.softmax(logits, dim=1)
#             prediction_probs.extend(probs)
#
#             # Get predictions
#             preds = torch.argmax(probs, dim=1)
#             predictions.extend(preds)
#             i += 1
#             if i == 10:
#                 break
#
#         predictions = torch.stack(predictions)
#         prediction_probs = torch.stack(prediction_probs)
#         true_labels = torch.stack(true_labels)
#         return predictions, prediction_probs, true_labels
