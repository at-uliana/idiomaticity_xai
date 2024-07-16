import torch


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
            'validation loss': []
        }
        for arg in args.__dict__:
            self.results[arg] = args.__dict__[arg]
        print("Done.")

    def train_batch(self, batch):
        self.model.train()
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.clone().detach()
        attention_mask = attention_mask.clone().detach()
        labels = labels.clone().detach()

        self.optimizer.zero_grad()

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        self.optimizer.step()
        return loss

    def fine_tune(self):
        val_loss = self.evaluate_model()
        print(f"Initial validation loss: {val_loss:.3f}")
        self.results['validation loss'].append(val_loss)

        print("--- Start fine-tuning ---")
        for epoch in range(self.args.n_epochs):
            print(f"--- Epoch {epoch+1}/{self.args.n_epochs} ---")
            i = 0
            for batch in self.train_loader:
                print(f"\tBatch {i+1}/{3}", end=' ')
                batch_loss = self.train_batch(batch)
                print(f"--- {batch_loss:.3f}")
                self.results['batch train loss'].append(batch_loss.item())
                i += 1
                if i == 3:
                    break
            print("Evaluate on the validation data set...")
            val_loss = self.evaluate_model()
            print(f"Validation loss: {val_loss:.3f}")
            self.results['validation loss'].append(val_loss)
            print("Done.")

    def evaluate_model(self):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            i = 0
            for batch in self.val_loader:
                input_ids, attention_mask, labels = batch
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                val_loss += loss.item()
                i += 1
                if i == 3:
                    break
        epoch_val_loss = val_loss / 3
        return epoch_val_loss


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
