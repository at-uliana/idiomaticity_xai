import torch
import json
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

class IdiomaticityTrainer:

    def __init__(self, model, optimizer, device, train_loader, val_loader, test_loader, args):
        print("Initializing the trainer...")
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
            'n test batches': len(test_loader)
        }
        for arg in args.__dict__:
            self.results[arg] = args.__dict__[arg]
        print("Done.")

    def train_batch(self, batch):
        self.model.train()
        self.model.to(self.device)
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.clone().detach()
        attention_mask = attention_mask.clone().detach()
        labels = labels.clone().detach()

        input_ids.to(self.device)
        attention_mask.to(self.device)
        labels.to(self.device)

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
        self.results['validation loss'].append(val_loss)
        self.results['validation accuracy'].append(val_acc)

        print("--- Start fine-tuning ---")
        for epoch in range(self.args.n_epochs):
            print('-------------------')
            print(f"Epoch {epoch+1}/{self.args.n_epochs}")
            print('-------------------')
            print(f"    Batch\t-\tLoss\t-\tAccuracy")
            i = 0
            for batch in self.train_loader:
                print(f"\t{i+1}/{3}", end='\t')
                batch_loss, batch_acc = self.train_batch(batch)
                print(f"\t-\t{batch_loss:.3f}", end='')
                print(f"\t-\t{batch_acc:.3f}")

                self.results['batch train loss'].append(batch_loss.item())
                self.results['batch train accuracy'].append(batch_acc)

                i += 1
                if i == 3:
                    break
            print("Evaluate on the validation data set...")
            val_loss, val_acc = self.evaluate_model()
            print(f"Validation loss: {val_loss:.3f}")
            print(f"Validation accuracy: {val_acc:.3f}")
            self.results['validation loss'].append(val_loss)
            self.results['validation accuracy'].append(val_acc)
            print("Done.")

    def evaluate_model(self):
        self.model.eval()
        validation_loss = 0.0
        validation_accuracy = 0.0

        with torch.no_grad():
            i = 0
            for batch in self.val_loader:
                input_ids, attention_mask, labels = batch
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss, logits = outputs.loss, outputs.logits
                validation_loss += loss.item()
                predictions = torch.argmax(torch.softmax(logits, dim=1), dim=1)
                accuracy = (predictions == labels).sum().item()/len(predictions)
                validation_accuracy += accuracy
                i += 1
                if i == 3:
                    break
        validation_loss = validation_loss / 3
        validation_accuracy = validation_accuracy / 3
        return validation_loss, validation_accuracy

    def save_config(self, path):
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
            i = 0
            for batch in self.test_loader:
                input_ids, attention_mask, labels = batch
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss, logits = outputs.loss, outputs.logits
                predictions = torch.argmax(torch.softmax(logits, dim=1), dim=1)
                predicted.extend(predictions.tolist())
                true.extend(labels.tolist())
                i += 1
                if i == 3:
                    break
        test_dict = {
            'predictions': predicted,
            'accuracy': accuracy_score(true, predicted),
            'f1-score': f1_score(true, predicted),
            'precision': precision_score(true, predicted),
            'recall': recall_score(true, predicted)
        }
        self.results['test'] = test_dict
        return test_dict


#
# # Train model on one batch
# def train_batch(model, batch, optimizer, device, args):
#     model.train()
#     input_ids, attention_mask, labels = batch
#     input_ids = input_ids.clone().detach()
#     attention_mask = attention_mask.clone().detach()
#     labels = labels.clone().detach()
#
#     optimizer.zero_grad()
#
#     outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
#     loss = outputs.loss
#     loss.backward()
#
#     torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#
#     optimizer.step()
#     return loss
#
#
# # Calculate validation loss
# def evaluate_model(model, validation_loader):
#     model.eval()
#     val_loss = 0.0
#     with torch.no_grad():
#         i = 0
#         for batch in validation_loader:
#             input_ids, attention_mask, labels = batch
#             outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
#             loss = outputs.loss
#             val_loss += loss.item()
#             i += 1
#             if i == 5:
#                 break
#     epoch_val_loss = val_loss / 5
#     return epoch_val_loss
#
#
# # Main training loop
# def train_loop(model, train_dataloader, validation_dataloader, epochs, optimizer, device, args):
#
#     results = {
#         'batch_loss': [],        # train loss calculated on each batch
#         'batch_acc': [],         # train accuracy calculated on each batch
#         'validation_loss': [],   # validation loss after each epoch
#         'validation_acc': []     # validation accuracy after each epoch
#     }
#
#     val_loss = evaluate_model(model, validation_dataloader)
#     print(f"Initial validation loss: {val_loss:.3f}")
#     results['validation_loss'].append(val_loss)
#
#     for epoch in range(epochs):
#         print(f"--- Epoch {epoch+1}/{epochs} ---")
#         i = 0
#         for batch in train_dataloader:
#             print(f"\tBatch {i+1}/{3}", end=' ')
#             batch_loss = train_batch(model, batch, optimizer, device, args)
#             print(f"--- {batch_loss:.3f}")
#             results['batch_loss'].append(batch_loss.item())
#             i += 1
#             if i == 3:
#                 break
#         print("Evaluate on the validation data set...")
#         val_loss = evaluate_model(model, validation_dataloader)
#         print(f"Validation loss: {val_loss:.3f}")
#         results['validation_loss'].append(val_loss)
#         print("Done.")
#
#     return results
#
