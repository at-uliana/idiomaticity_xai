import torch
import json
import os
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from utils import are_all_model_parameters_on_gpu, are_all_model_buffers_on_gpu


def make_model_name():
    name = datetime.now().strftime("model %d-%m-%Y %H:%M")
    return name


class IdiomaticityTrainer:

    def __init__(self, model, optimizer, device, train_loader, val_loader, test_loader, n_epochs,
                 save_checkpoints=False, model_name=None, output_dir='checkpoints'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.device = device
        self.n_epochs = n_epochs
        self.save_checkpoints = save_checkpoints
        self.output_dir = output_dir
        if model_name is None:
            self.model_name = make_model_name()
        else:
            self.model_name = model_name
        self.validation_loss = []
        self.validation_accuracy = []

    def train_batch(self, batch):
        self.model.train()

        # Send batch to gpu
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        labels = labels.to(self.device)

        # Zero out gradients
        self.optimizer.zero_grad()

        # Prepare batch
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Calculate training accuracy
        logits = outputs.logits
        predictions = torch.argmax(torch.softmax(logits, dim=1), dim=1)
        accuracy = (predictions == labels).sum().item() / len(predictions)

        # Backward
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        # Optimizer step
        self.optimizer.step()
        return loss, accuracy

    def fine_tune(self):
        # Calculate initial loss
        val_loss, val_acc = self.evaluate_model()
        self.validation_loss.append(val_loss)
        self.validation_accuracy.append(val_acc)

        print()
        print("-----------")
        print("Fine-tuning")
        print("-----------")

        for epoch in range(self.n_epochs):
            print(f" - Epoch {epoch + 1} out of {self.n_epochs}")

            i = 0
            for batch in self.train_loader:
                _ = self.train_batch(batch)
                i += 1
                if i == 2:
                    break

            # Calculate validation loss and accuracy
            val_loss, val_acc = self.evaluate_model()
            print(f"\tValidation loss: {val_loss:.3f}")
            print(f"\tValidation accuracy: {val_acc:.3f}")

            # Save validation loss and accuracy
            self.validation_loss.append(val_loss)
            self.validation_accuracy.append(val_acc)

            # Save the model
            if self.save_checkpoints:
                # Save model after each checkpoint
                path = os.path.join(self.output_dir, self.model_name + f" epoch={epoch}.pt")
                self.save_model(path)
            else:
                # Save only if it's the last epoch
                if epoch == self.n_epochs - 1:
                    path = os.path.join(self.output_dir, self.model_name + ".pt")
                    self.save_model(path)
            print()

    def evaluate_model(self):
        self.model.eval()
        validation_loss = 0.0
        validation_accuracy = 0.0

        with torch.no_grad():
            i = 0
            for batch in self.val_loader:

                # Extract batch and send to GPU
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)

                # Evaluate on batch, calculate loss and accuracy
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss, logits = outputs.loss, outputs.logits
                validation_loss += loss.item()
                predictions = torch.argmax(torch.softmax(logits, dim=1), dim=1)
                accuracy = (predictions == labels).sum().item() / len(predictions)
                validation_accuracy += accuracy
                i += 1
                if i == 2:
                    break

        validation_loss = validation_loss / len(self.val_loader)
        validation_accuracy = validation_accuracy / len(self.val_loader)
        return validation_loss, validation_accuracy

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def save_config(self):

        output = {
            'validation loss': self.validation_loss,
            'validation accuracy': self.validation_accuracy,
            'epochs': self.n_epochs,
            'model name': self.model_name,

        }

        path = os.path.join(self.output_dir, "output.json")
        json.dump(output, open(path, 'w'), indent=True)

    # def test_model(self):
    #     self.model.eval()
    #     test_accuracy = 0.0
    #     test_loss = 0.0
    #     predicted = []
    #     true = []
    #
    #     with torch.no_grad():
    #         for batch in self.test_loader:
    #             input_ids, attention_mask, labels = batch
    #             outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    #             loss, logits = outputs.loss, outputs.logits
    #             predictions = torch.argmax(torch.softmax(logits, dim=1), dim=1)
    #             predicted.extend(predictions.tolist())
    #             true.extend(labels.tolist())
    #     test_dict = {
    #         'predictions': predicted,
    #         'accuracy': accuracy_score(true, predicted),
    #         'f1-score': f1_score(true, predicted),
    #         'precision': precision_score(true, predicted),
    #         'recall': recall_score(true, predicted)
    #     }
    #     self.results['test'] = test_dict
    #     return test_dict
    #
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
