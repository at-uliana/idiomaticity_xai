import torch


def train_batch(model, batch, optimizer, args):
    model.train()
    input_ids, attention_mask, labels = batch
    input_ids = input_ids.clone().detach()
    attention_mask = attention_mask.clone().detach()
    labels = labels.clone().detach()

    optimizer.zero_grad()

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()
    return loss

#
# def train(model, train_dataloader, optimizer, epochs, device):


def evaluate_model(model, validation_loader):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        i = 0
        for batch in validation_loader:
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            val_loss += loss.item()
            i += 1
            if i == 30:
                break
    epoch_val_loss = val_loss / 30
    return epoch_val_loss
