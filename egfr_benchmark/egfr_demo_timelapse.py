import os
import sys
sys.path.append('../release')

import torch
import torch.nn as nn
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
use_cuda = torch.cuda.is_available()

import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import pickle
from rdkit import Chem, DataStructs
from rdkit.Chem import PandasTools
from sklearn.ensemble import RandomForestClassifier as RFC

from stackRNN import StackAugmentedRNN
from data import GeneratorData
from utils import canonical_smiles, get_fp
from predictor import VanillaQSAR
from reinforcement import Reinforcement

import time
from analysis_utils import compare_libraries

# Consider: moving more code to module scope

# Util functions
def estimate_and_update(generator, predictor, n_to_generate, threshold=None, batch_size=16,
                        plot_counts=False, plot=True, return_metrics=False, **kwargs):
    generated = []
    pbar = tqdm(range(n_to_generate))
    for i in pbar:
        pbar.set_description("Generating molecules...")
        generated += generator.evaluate(gen_data, predict_len=120, batch_size=batch_size)
    
    sanitized = canonical_smiles(generated, sanitize=False, throw_warning=False)[:-1]
    unique_smiles, counts = np.unique(sanitized, return_counts=True)
    unique_smiles, counts = list(unique_smiles)[1:], counts[1:]
    
    if plot_counts:
        if plot:
            plt.hist(counts)
            plt.gca().set_yscale('log')
            plt.title('Distribution of counts of generated smiles')
            plt.xlabel('Counts observed')
            plt.show()
        max_counts = max(counts)
        if max_counts > 1:
            print('Trajectories with max counts:')
            for i in np.where(counts == max_counts)[0]:
                print('%d\t%s' % (counts[i], unique_smiles[i]))
    smiles, prediction, nan_smiles = predictor.predict(unique_smiles, get_features=get_fp)  
    plot_hist(prediction, len(generated), threshold, plot=plot)
    valid_fraction = len(prediction) / len(generated)
    active_fraction = np.mean(prediction >= threshold)
    metrics = {'valid_fraction': valid_fraction,
               'active_fraction': active_fraction}
    if plot_counts:
        metrics['max_counts'] = max_counts
    if return_metrics:
        return smiles, prediction, metrics
    else:
        return smiles, prediction

def plot_hist(prediction, n_to_generate, threshold=None, plot=True):
    print("Mean value of predictions:", prediction.mean())
    print("Proportion of valid SMILES:", len(prediction)/n_to_generate)
    #ax = sns.kdeplot(prediction, shade=True)
    if plot:
        ax = sns.distplot(prediction, kde=False)
        if threshold is not None:
            ax.axvline(x=threshold, color="red")
        ax.set(xlabel='Predicted pIC50', 
            title='Distribution of predicted pIC50 for generated molecules')
        plt.show()

def simple_moving_average(prev_values, new_value, ma_window_size=10):
    ma_value = sum(prev_values[-(ma_window_size-1):]) + new_value
    ma_value = ma_value / (len(prev_values[-(ma_window_size-1):]) + 1.)
    return ma_value

# replay_data_path - filename of file to instantiate replay buffer and fine-tuning instances
# primed_path - checkpoint model to initialize the generator
def main(n_iterations=20,
         n_policy=10,
         n_policy_replay=15,
         batch_size=16,
         n_fine_tune=None,
         seed=None,
         replay_data_path='../data/gen_actives.smi',
         primed_path='../checkpoints/generator/checkpoint_batch_training',
         save_every=2,
         save_path=None):
    save_path = os.path.splitext(save_path)[0]
    save_path = save_path.split('-')[0]
    if n_fine_tune is None:
        n_fine_tune = n_iterations
    
    # initialize RNG seeds for reproducibility
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    gen_data_path = '../data/chembl_22_clean_1576904_sorted_std_final.smi'
    tokens = [' ', '<', '>', '#', '%', ')', '(', '+', '-', '/', '.', '1', '0', '3', '2', '5', '4', '7',
              '6', '9', '8', '=', 'a', '@', 'C', 'B', 'F', 'I', 'H', 'O', 'N', 'P', 'S', '[', ']',
              '\\', 'c', 'e', 'i', 'l', 'o', 'n', 'p', 's', 'r']
    global gen_data;
    gen_data = GeneratorData(gen_data_path, delimiter='\t',
                             cols_to_read=[0], keep_header=True, tokens=tokens)

    # Setting up the generative model
    hidden_size = 1500
    stack_width = 1500
    stack_depth = 200
    layer_type = 'GRU'
    optimizer = torch.optim.SGD
    lr = 0.0002
    generator = StackAugmentedRNN(input_size=gen_data.n_characters, hidden_size=hidden_size,
                                         output_size=gen_data.n_characters,
                                         layer_type=layer_type, n_layers=1, is_bidirectional=False,
                                         has_stack=True, stack_width=stack_width, stack_depth=stack_depth,
                                         use_cuda=use_cuda,
                                         optimizer_instance=optimizer, lr=lr)
    # Use a model pre-trained on active molecules 
    generator.load_model(primed_path)

    # Setting up the predictor
    model_instance = RFC
    model_params = {'n_estimators': 250, 'n_jobs': 10}
    predictor = VanillaQSAR(model_instance=model_instance,
                            model_params=model_params,
                            model_type='classifier')
    predictor.load_model('../checkpoints/predictor/egfr_rfc')
    
    # Setting up the reinforcement model
    def get_reward(smiles, predictor, threshold, invalid_reward=1.0, get_features=get_fp):
        mol, prop, nan_smiles = predictor.predict([smiles], get_features=get_features)
        if len(nan_smiles) == 1:
            return invalid_reward
        if prop[0] >= threshold:
            return 10.0
        else:
            return invalid_reward

    RL_model = Reinforcement(generator, predictor, get_reward)
    
    # Define replay update functions
    def update_threshold(cur_threshold, prediction, proportion=0.15, step=0.05):
        if (prediction >= cur_threshold).mean() >= proportion:
            new_threshold = min(cur_threshold + step, 1.0)
            return new_threshold
        else:
            return cur_threshold

    def update_data(smiles, prediction, replay_data, fine_tune_data, threshold):
        for i in range(len(prediction)):
            if prediction[i] >= max(threshold, 0.2):
                fine_tune_data.file.append('<' + smiles[i] + '>')
            if prediction[i] >= threshold:    
                replay_data.append(smiles[i])
        return fine_tune_data, replay_data
    
    fine_tune_data = GeneratorData(replay_data_path,
                                   tokens=tokens,
                                   cols_to_read=[0],
                                   keep_header=True)
    replay_data = GeneratorData(replay_data_path,
                                tokens=tokens,
                                cols_to_read=[0],
                                keep_header=True)
    replay_data = [traj[1:-1] for traj in replay_data.file]

    rl_losses = []
    rewards = []
    n_to_generate = 200
    threshold = 0.05
    start = time.time()
    active_threshold = 0.75
    
    tmp = sys.stdout
    sys.stdout = sys.__stdout__
    smiles, predictions, gen_metrics = estimate_and_update(RL_model.generator,
                                                           RL_model.predictor, 
                                                           1000,
                                                           batch_size=batch_size,
                                                           plot=False,
                                                           threshold=active_threshold,
                                                           return_metrics=True)
    sys.stdout = tmp
    mol_data = pd.DataFrame(dict(smiles=smiles, predictions=predictions))
    if save_path:
        save_path_ = save_path + '-0.smi'
        mol_data.to_csv(save_path_, index=False, header=False)

      #  log_path = save_path + '.log'
      #  with open(log_path, 'wt') as f:
      #      print('starting log', file=f)
    
    for i in range(n_iterations):
        print('%3.d Training on %d replay instances...' % (i+1, len(replay_data)))
        print('Setting threshold to %f' % threshold)
        
        print('Policy gradient...')
        for j in trange(n_policy, desc=' %3.d Policy gradient...' % (i+1)):
            cur_reward, cur_loss = RL_model.policy_gradient(gen_data, get_features=get_fp, 
	        					  threshold=threshold)
	    
            rewards.append(simple_moving_average(rewards, cur_reward)) 
            rl_losses.append(simple_moving_average(rl_losses, cur_loss))
        print('Loss: %f' % rl_losses[-1])
        print('Reward: %f' % rewards[-1])
        smiles_cur, prediction_cur = estimate_and_update(RL_model.generator, 
							 RL_model.predictor, 
							 n_to_generate,
                                                         batch_size=batch_size,
							 get_features=get_fp,
							 threshold=active_threshold,
                                                         plot_counts=True,
                                                         plot=False)
        fine_tune_data, replay_data = update_data(smiles_cur, prediction_cur, replay_data,
						  fine_tune_data, threshold)
        threshold = update_threshold(threshold, prediction_cur)
        print('Sample trajectories:')
        for sm in smiles_cur[:5]:
            print(sm)
            
        print('Policy gradient replay...')
        for j in trange(n_policy_replay, desc='%3.d Policy gradient replay...' % (i+1)):
            cur_reward, cur_loss = RL_model.policy_gradient(gen_data, get_features=get_fp, 
							  replay=True, replay_data=replay_data, 
							  threshold=threshold)
        smiles_cur, prediction_cur = estimate_and_update(RL_model.generator, 
							 RL_model.predictor, 
							 n_to_generate,
                                                         batch_size=batch_size,
							 get_features=get_fp,
							 threshold=active_threshold,
                                                         plot=False)
        fine_tune_data, replay_data = update_data(smiles_cur, prediction_cur, replay_data,
						  fine_tune_data, threshold)
        threshold = update_threshold(threshold, prediction_cur)
        print('Sample trajectories:')
        for sm in smiles_cur[:5]:
            print(sm)
	
        print('Fine tuning...')
        RL_model.fine_tune(data=fine_tune_data, n_steps=n_fine_tune, batch_size=batch_size, print_every=10000)
        smiles_cur, prediction_cur = estimate_and_update(RL_model.generator, 
							 RL_model.predictor, 
							 n_to_generate,
                                                         batch_size=batch_size,
							 get_features=get_fp,
							 threshold=active_threshold,
                                                         plot=False)
        fine_tune_data, replay_data = update_data(smiles_cur, prediction_cur, replay_data,
						  fine_tune_data, threshold)
        threshold = update_threshold(threshold, prediction_cur)
        print('Sample trajectories:')
        for sm in smiles_cur[:5]:
            print(sm)
        print('')

        if (i+1) % save_every == 0:
            # redirect output to keep valid log
            tmp = sys.stdout
            sys.stdout = sys.__stdout__
            smiles, predictions, gen_metrics = estimate_and_update(RL_model.generator,
                                                                   RL_model.predictor, 
                                                                   1000,
                                                                   batch_size=batch_size,
                                                                   plot=False,
                                                                   threshold=active_threshold,
                                                                   return_metrics=True)
            mol_data = pd.DataFrame(dict(smiles=smiles, predictions=predictions))
            if save_path:
                save_path_ = save_path + '-%d.smi' % (i+1)
                mol_data.to_csv(save_path_, index=False, header=False)
            sys.stdout = tmp

    duration = time.time() - start
    train_metrics = {}
    train_metrics['duration'] = duration
    mol_actives = mol_data[mol_data.predictions > active_threshold]
    egfr_data = pd.read_csv('../data/egfr_with_pubchem.csv')
    egfr_actives = egfr_data[egfr_data.predictions > active_threshold]
    mol_actives['molecules'] = mol_actives.smiles.apply(Chem.MolFromSmiles)
    egfr_actives['molecules'] = egfr_actives.smiles.apply(Chem.MolFromSmiles)
    lib_metrics = compare_libraries(mol_actives, egfr_actives, properties=['MolWt', 'MolLogP'],
                                    return_metrics=True, plot=False)
    # collate results of training
    results = {}
    results.update(train_metrics)
    results.update(gen_metrics)
    results.update(lib_metrics)

    params = dict(n_iterations=n_iterations,
                  n_policy=n_policy,
                  n_policy_replay=n_policy_replay,
                  n_fine_tune=n_fine_tune,
                  seed=seed,
                  replay_data_path=replay_data_path,
                  primed_path=primed_path)
    if save_path is not None:
        results['save_path'] = save_path_
    print('Metrics for %s:' % params)
    print(results)

if __name__ == '__main__':
    args = sys.argv[1:]
    kwargs = {}
    for arg in args:
        key, val = arg.split('=')
        try:
            val = int(val)
        except:
            pass
        kwargs[key] = val
    main(**kwargs)
