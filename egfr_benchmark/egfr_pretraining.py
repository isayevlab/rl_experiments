# 02/12/2021
# code to generate initial pre-loaded models
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
sys.path.append('../release')

import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import pickle
from rdkit import Chem, DataStructs

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, ExponentialLR
import torch.functional as F
use_cuda = torch.cuda.is_available()

from data import GeneratorData, PredictorData
from stackRNN import StackAugmentedRNN
from utils import get_fp, canonical_smiles
from reinforcement import Reinforcement

from sklearn.ensemble import RandomForestClassifier as RFC
from predictor import VanillaQSAR


# training the predictor
pred_data = PredictorData('../data/egfr_with_pubchem.csv', get_features=get_fp)
model_instance = RFC
model_params = {'n_estimators': 250,
                'n_jobs': 10}
np.random.seed(42)
my_predictor = VanillaQSAR(model_instance=model_instance,
                           model_params=model_params,
                           ensemble_size=10)
try:
    my_predictor.load_model('../checkpoints/predictor/egfr_rfc')
except:
    my_predictor.fit_model(pred_data, cv_split='random')
    my_predictor.save_model('../checkpoints/predictor/egfr_rfc')

# pretraining the generator
print('Pretraining generator...')
np.random.seed(42)
torch.manual_seed(42)
gen_data_path = '../data/chembl_22_clean_1576904_sorted_std_final.smi'
tokens = [' ', '<', '>', '#', '%', ')', '(', '+', '-', '/', '.', '1', '0', '3', '2', '5', '4', '7',
          '6', '9', '8', '=', 'A', '@', 'C', 'B', 'F', 'I', 'H', 'O', 'N', 'P', 'S', '[', ']',
          '\\', 'c', 'e', 'i', 'l', 'o', 'n', 'p', 's', 'r']

gen_data = GeneratorData(training_data_path=gen_data_path, delimiter='\t',
                         cols_to_read=[0], keep_header=True, tokens=tokens)
hidden_size = 1500
stack_width = 1500
stack_depth = 200
layer_type = 'GRU'
lr = 0.001
optimizer = torch.optim.Adadelta
my_generator_max = StackAugmentedRNN(input_size=gen_data.n_characters, hidden_size=hidden_size,
                                     output_size=gen_data.n_characters,
                                     layer_type=layer_type, n_layers=1, is_bidirectional=False,
                                     has_stack=True, stack_width=stack_width, stack_depth=stack_depth,
                                     use_cuda=use_cuda,
                                     lr=lr, optimizer_instance=optimizer)
model_path = '../checkpoints/generator/checkpoint_batch_training'

batch_size = 16
try:
    my_generator_max.load_model(model_path)
except:
    losses = my_generator_max.fit(gen_data, batch_size, 1500000)
    my_generator_max.save_model(model_path)
    with open('losses.txt','wt') as f:
        for val in losses:
            print(val, file=f)

# fine-tuning the generator on pre-existing libraries
def get_reward_max(smiles, predictor, threshold, invalid_reward=1.0, get_features=get_fp):
    mol, prop, nan_smiles = predictor.predict([smiles], get_features=get_features)
    if len(nan_smiles) == 1:
        return invalid_reward
    if prop[0] >= threshold:
        return 10.0
    else:
        return invalid_reward

RL_max = Reinforcement(my_generator_max, my_predictor, get_reward_max)

n_iterations = 60000
data_path = ['../data/egfr_actives.smi',
             '../data/egfr_enamine.smi',
             '../data/egfr_mixed.smi']
save_path = ['../checkpoints/generator/egfr_clf_rnn_primed',
             '../checkpoints/generator/egfr_clf_rnn_enamine_primed',
             '../checkpoints/generator/egfr_clf_rnn_mixed_primed']
for dpath, mpath in zip(data_path, save_path):
    print('Fine-tuning on %s...' % dpath)
    np.random.seed(42)
    torch.manual_seed(42)

    actives_data = GeneratorData(dpath,
                                 tokens=tokens,
                                 cols_to_read=[0],
                                 keep_header=True)
    RL_max.generator.load_model(model_path)
    RL_max.fine_tune(data=actives_data, n_steps=n_iterations, batch_size=batch_size)
    RL_max.generator.save_model(mpath)
