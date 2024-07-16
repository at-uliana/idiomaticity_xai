import pandas as pd


class Args:

    def __init__(self, batch_size=8, n_epochs=5, learning_rate=1e+5, freeze_pretrained=False):
        self.batch_size=batch_size
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.freeze_pretrained = freeze_pretrained


def train_test_dev_split(data_path, split_path, cols=None):
    if cols is None:
        cols = ['idiom', 'sentence', 'label', 'transparency', 'head pos', 'corpus', 'split']
    data = pd.read_csv(data_path, sep='\t')
    split = pd.read_csv(split_path, sep='\t')
    data['split'] = split['split']
    train_data = data[data['split'] == 'train']
    test_data = data[data['split'] == 'test']
    dev_data = data[data['split'] == 'dev']
    return train_data[cols], dev_data[cols], test_data[cols]
#
#
# num_epochs = 1  # Only a few epochs for the test
# total_steps = len(train_dataloader) * num_epochs
# scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
#
# # Training loop for the learning rate range test
# model.train()
# learning_rates = []
# losses = []
#
# for batch in train_dataloader:
#     inputs, labels = batch
#     optimizer.zero_grad()
#
#     outputs = model(**inputs)
#     loss = criterion(outputs.logits, labels)
#
#     loss.backward()
#     optimizer.step()
#     scheduler.step()
#
#     # Record the learning rate and loss
#     learning_rates.append(optimizer.param_groups[0]["lr"])
#     losses.append(loss.item())
#
#     # Update the learning rate
#     for param_group in optimizer.param_groups:
#         param_group['lr'] *= 1.1  # Increase learning rate by a factor
#
# # Plot the loss vs. learning rate
# plt.plot(learning_rates, losses)
# plt.xscale('log')
# plt.xlabel('Learning Rate')
# plt.ylabel('Loss')
# plt.title('Learning Rate Range Test')
# plt.show()

