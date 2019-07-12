import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import csv
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('titanic/train.csv')
testset = pd.read_csv('titanic/test.csv')

print(dataset('Name'))
