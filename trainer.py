import torch
import json
import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from utils import are_all_model_parameters_on_gpu, are_all_model_buffers_on_gpu


class IdiomaticityTrainer:

    def __init__(self, model, optimizer, device, train_loader, val_loader, test_loader, args):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.device = device
        self.args = args
        self.results = {
            'batch train loss': [],
            'batch train accuracy': [],
            'validation loss': [],
            'validation accuracy': [],
            'n train batches': len(train_loader),
            'n val batches': len(val_loader),
            'n test batches': len(test_loader),
            'device': self.device
        }

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
        print(f"Initial validation loss: {val_loss:.3f}")
        print(f"Initial validation accuracy: {val_acc:.3f}")
        print()
        self.results['validation loss'].append(val_loss)
        self.results['validation accuracy'].append(val_acc)

        print("--- Start fine-tuning ---")
        print()
        for epoch in range(self.args.n_epochs):
            print('-------------------')
            print(f"Epoch {epoch}/{self.args.n_epochs}")
            print('-------------------')
            print(f"    Batch\t-\tLoss\t-\tAccuracy")
            for i, batch in enumerate(self.train_loader):
                print(f"\t{i+1}", end='\t')
                batch_loss, batch_acc = self.train_batch(batch)
                print(f"\t-\t{batch_loss:.3f}", end='')
                print(f"\t-\t{batch_acc:.3f}")

                self.results['batch train loss'].append(batch_loss.item())
                self.results['batch train accuracy'].append(batch_acc)

            # Save model after each epoch:
            if self.args.save_checkpoints:
                path = os.path.join(self.args.model_dir, self.args.model_name + f" e{epoch}.pt")
                print(f"Saving checkpoint to {path}.")
                self.save_model(path)
            else:
                # if this is the last epoch:
                if epoch == self.args.n_epochs - 1:
                    path = os.path.join(self.args.model_dir, self.args.model_name + '.pt')
                    print(f"Saving final model to {path}.")
                    self.save_model(path)

            val_loss, val_acc = self.evaluate_model()
            print(f"Validation loss: {val_loss:.3f}")
            print(f"Validation accuracy: {val_acc:.3f}")
            self.results['validation loss'].append(val_loss)
            self.results['validation accuracy'].append(val_acc)
            print("Done.")
            print()

    def evaluate_model(self):
        print("All parameters on GPU:", are_all_model_parameters_on_gpu(self.model))
        print("All buffers on GPU:", are_all_model_buffers_on_gpu(self.model))
        print()

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

    def save_config(self):
        path = os.path.join(self.args.output_dir, self.args.model_name + " results.json")
        json.dump(self.results, open(path, 'w'), indent=True)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def test_model(self):
        self.model.eval()
        test_accuracy = 0.0
        test_loss = 0.0
        predicted = []
        true = []

        with torch.no_grad():
            for batch in self.test_loader:
                input_ids, attention_mask, labels = batch
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss, logits = outputs.loss, outputs.logits
                predictions = torch.argmax(torch.softmax(logits, dim=1), dim=1)
                predicted.extend(predictions.tolist())
                true.extend(labels.tolist())
        test_dict = {
            'predictions': predicted,
            'accuracy': accuracy_score(true, predicted),
            'f1-score': f1_score(true, predicted),
            'precision': precision_score(true, predicted),
            'recall': recall_score(true, predicted)
        }
        self.results['test'] = test_dict
        return test_dict

    def get_predictions(self, external_dataloader=None):
        if external_dataloader:
            test_loader = external_dataloader
        else:
            test_loader = self.test_loader

        self.model.eval()
        predictions = []
        prediction_probs = []
        true_labels = []

        with torch.no_grad():
            i = 0
            for batch in test_loader:
                input_ids, attention_mask, labels = batch
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss, logits = outputs.loss, outputs.logits

                # Save `true` labels
                true_labels.extend(labels)

                # Get probabilities
                probs = torch.softmax(logits, dim=1)
                prediction_probs.extend(probs)

                # Get predictions
                preds = torch.argmax(probs, dim=1)
                predictions.extend(preds)
                i += 1
                if i == 10:
                    break

            predictions = torch.stack(predictions)
            prediction_probs = torch.stack(prediction_probs)
            true_labels = torch.stack(true_labels)
            return predictions, prediction_probs, true_labels
