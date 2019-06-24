import torch
from torch.utils import data
import pickle
import h5py
import numpy as np
import glob

class CompressDataset(data.Dataset):
  # Characterizes a dataset for PyTorch
  def __init__(self, filename= 'tensor_data'):
        'Initialization'
        self.filename = filename
        self.f = h5py.File(filename, 'r')

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.f.keys())

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        one_data_sample = np.array(self.f[str(index)])
        return one_data_sample


class TensorDataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, rootpath= 'tensor_data', file_format='.h5'):
        'Initialization'
        self.file = rootpath
        self.file_format = file_format
        self.file_list = glob.glob(self.rootpath + '/*' + self.file_format)

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.file_list)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        one_data_sample = pickle.load(open(self.file_list[index],'rb'))
        return one_data_sample

  def __create_random_dataset__(self):
      for index in range(100):
        data_tensor = np.random.normal(0, 1, (18, 32, 32))
        pickle.dump(data_tensor, open("tensor_data/sample_{}.p".format(index),"wb"))

