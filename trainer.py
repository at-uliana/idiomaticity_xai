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

    def __init__(self, model, optimizer, device, n_epochs, train_loader, val_loader,
                 model_name=None, output_dir='checkpoints'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.n_epochs = n_epochs
        self.output_dir = output_dir
        if model_name is None:
            self.model_name = make_model_name()
        else:
            self.model_name = model_name
        self.validation_loss = []
        self.validation_accuracy = []
        self.best_model = None
        self.best_epoch = None

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

            for batch in self.train_loader:
                _ = self.train_batch(batch)

            # Calculate validation loss and accuracy
            val_loss, val_acc = self.evaluate_model()
            print(f"\tValidation loss: {val_loss:.3f}")
            print(f"\tValidation accuracy: {val_acc:.3f}")

            # Extract the validation loss of the previous epoch
            previous_loss = self.validation_loss[-1]

            # add validation loss and accuracy
            self.validation_loss.append(val_loss)
            self.validation_accuracy.append(val_acc)

            # Stop training if validation loss is going up
            if val_loss > previous_loss:
                print("Early stopping. The checkpoint will not be saved.")
                break
            else:
                # Save the model
                print("Saving the checkpoint.")
                name = self.model_name + f" epoch={epoch}.pt"
                path = os.path.join(self.output_dir, name)
                self.save_model(path)
                self.best_model = name
                self.best_epoch = epoch
                print()

                # # Save the model
                # if self.save_checkpoints:
                #     # Save model after each checkpoint
                #     path = os.path.join(self.output_dir, self.model_name + f" epoch={epoch}.pt")
                #     self.save_model(path)
                # else:
                #     # Save only if it's the last epoch
                #     if epoch == self.n_epochs - 1:
                #         path = os.path.join(self.output_dir, self.model_name + ".pt")
                #         self.save_model(path)

    def evaluate_model(self):
        self.model.eval()
        validation_loss = 0.0
        validation_accuracy = 0.0

        with torch.no_grad():

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
            'best model': self.best_model
        }

        path = os.path.join(self.output_dir, "output.json")
        json.dump(output, open(path, 'w'), indent=True)

