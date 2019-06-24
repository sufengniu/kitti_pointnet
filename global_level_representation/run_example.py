
import torch
import torch.utils.data as data

from my_dataset import TensorDataset

train_set = TensorDataset(rootpath='tensor_data', file_format='.p')

train_loader = data.DataLoader(
    dataset=train_set, batch_size=4, shuffle=True, num_workers=1)

for batch, data in enumerate(train_loader):
	print("batch: {}; data: {}".format(batch, data.shape))