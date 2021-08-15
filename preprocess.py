from Slice import Slice
from hparam import hparams as hp
import os

source_train_dir = './dataset/train/source'
label_train_dir = './dataset/train/label'

if not os.path.exists(hp.source_train_dir):
    os.makedirs(hp.source_train_dir)
if not os.path.exists(hp.label_train_dir):
    os.makedirs(hp.label_train_dir)

files = os.listdir(source_train_dir)
for file in files:
    Slice(source_train_dir + file, hp.source_train_dir)
files = os.listdir(label_train_dir)
for file in files:
    Slice(label_train_dir + file, hp.label_train_dir)