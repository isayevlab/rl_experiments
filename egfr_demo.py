import os
import sys
from functools import partial

import torch
import torch.nn as nn
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
use_cuda = torch.cuda.is_available()

import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import pickle
from rdkit import Chem, DataStructs
from rdkit.Chem import PandasTools
from sklearn.ensemble import RandomForestClassifier as RFC

sys.path.append('./release')
from stackRNN import StackAugmentedRNN
from data import GeneratorData
from utils import canonical_smiles, get_fp
from predictor import VanillaQSAR
from reinforcement import Reinforcement

import time
from analysis_utils import compare_libraries


# Consider: moving more code to module scope
#gen_data_path = './data/chembl_22_clean_1576904_sorted_std_final.smi'
#tokens = [' ', '<', '>', '#', '%', ')', '(', '+', '-', '/', '.', '1', '0', '3', '2', '5', '4', '7',
#          '6', '9', '8', '=', 'a', '@', 'C', 'B', 'F', 'I', 'H', 'O', 'N', 'P', 'S', '[', ']',
#          '\\', 'c', 'e', 'i', 'l', 'o', 'n', 'p', 's', 'r']
#gen_data = GeneratorData(gen_data_path, delimiter='\t',
#                         cols_to_read=[0], keep_header=True, tokens=tokens)


# Util functions
def estimate_and_update(generator, predictor, n_to_generate, threshold=None, 
                        plot_counts=False, plot=True, return_metrics=False, **kwargs):
    generated = []
    pbar = tqdm(range(n_to_generate))
    for i in pbar:
        pbar.set_description("Generating molecules...")
        generated += generator.evaluate(gen_data, predict_len=120, batch_size=16)
    
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
        if max(counts) > 1:
            print('Trajectories with max counts:')
            for i in np.where(counts == max(counts))[0]:
                print('%d\t%s' % (counts[i], unique_smiles[i]))
    smiles, prediction, nan_smiles = predictor.predict(unique_smiles, get_features=get_fp)  
    plot_hist(prediction, len(generated), threshold, plot=plot)
    valid_fraction = len(prediction) / len(generated)
    active_fraction = np.mean(prediction >= threshold)
    metrics = {'valid_fraction': valid_fraction,
               'active_fraction': active_fraction}
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
def main(**kwargs):
    params = dict(n_iterations=20,
                  n_policy=10,
                  n_policy_replay=15,
                  n_fine_tune=None,
                  replay_data_path='./data/replay_data.smi',
                  primed_path= './checkpoints/generator/checkpoint_batch_training',
                  save_path=None)
    params.update(kwargs)
    # couple fine tuning with n_iterations by default
    if params['n_fine_tune'] is None:
        params['n_fine_tune'] = params['n_iterations']

    gen_data_path = './data/chembl_22_clean_1576904_sorted_std_final.smi'
    tokens = [' ', '<', '>', '#', '%', ')', '(', '+', '-', '/', '.', '1', '0', '3', '2', '5', '4', '7',
              '6', '9', '8', '=', 'a', '@', 'C', 'B', 'F', 'I', 'H', 'O', 'N', 'P', 'S', '[', ']',
              '\\', 'c', 'e', 'i', 'l', 'o', 'n', 'p', 's', 'r']
    global gen_data;
    gen_data = GeneratorData(gen_data_path, delimiter='\t',
                             cols_to_read=[0], keep_header=True, tokens=tokens)

    # Loading the predictor
    #model_instance = RFC
    #model_params = {'n_estimators': 200,
    #                'min_samples_leaf': 2}
    #predictor = VanillaQSAR(model_instance=model_instance, model_params=model_params)
    #predictor.load_model('../project/checkpoints/predictor/cdk1_rfc_augmented')

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
    #model_path = './checkpoints/generator/checkpoint_batch_training'
    #generator.load_model(model_path)
    # Faster: use a model pre-trained on active molecules 
    primed_path = params['primed_path']
    generator.load_model(primed_path)

    # Setting up the predictor
    model_instance = RFC
    model_params = {'n_estimators': 250, 'n_jobs': 10}
    predictor = VanillaQSAR(model_instance=model_instance,
                            model_params=model_params,
                            model_type='classifier')
    predictor.load_model('../project/checkpoints/predictor/egfr_rfc')
    
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
    
    replay_data_path = params['replay_data_path']
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
    n_iterations = params['n_iterations']
    n_policy = params['n_policy']
    n_policy_replay = params['n_policy_replay']
    n_fine_tune = params['n_fine_tune']
    save_path = params['save_path']
    active_threshold = 0.75

    for i in range(n_iterations):
        print('%3.d Training on %d replay instances...' % (i+1, len(replay_data)))
        print('Setting threshold to %f' % threshold)
        print('Policy gradient...')
        for j in trange(n_policy, desc=' %3.d Policy gradient...' % (i+1)):
            cur_reward, cur_loss = RL_model.policy_gradient(gen_data, get_features=get_fp, 
	        					  threshold=threshold)
	    
            rewards.append(simple_moving_average(rewards, cur_reward)) 
            rl_losses.append(simple_moving_average(rl_losses, cur_loss))
        # delete after use
        _, pred_buff, _ = RL_model.predictor.predict(replay_data, 
                                                     get_features=get_fp)
        print('Mean activity in replay buffer: %f' % pred_buff.mean())
        print('Loss: %f' % rl_losses[-1])
        print('Reward: %f' % rewards[-1])
        smiles_cur, prediction_cur = estimate_and_update(RL_model.generator, 
							 RL_model.predictor, 
							 n_to_generate,
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
            
#	plt.plot(rewards)
#	plt.xlabel('Training iteration')
#	plt.ylabel('Average reward')
#	plt.show()
#	plt.plot(rl_losses)
#	plt.xlabel('Training iteration')
#	plt.ylabel('Loss')
#	plt.show()
        print('Policy gradient replay...')
        for j in trange(n_policy_replay, desc='%3.d Policy gradient replay...' % (i+1)):
            cur_reward, cur_loss = RL_model.policy_gradient(gen_data, get_features=get_fp, 
							  replay=True, replay_data=replay_data, 
							  threshold=threshold)
	    
        smiles_cur, prediction_cur = estimate_and_update(RL_model.generator, 
							 RL_model.predictor, 
							 n_to_generate,
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
        RL_model.fine_tune(data=fine_tune_data, n_steps=n_fine_tune, batch_size=16, print_every=10000)
	
        smiles_cur, prediction_cur = estimate_and_update(RL_model.generator, 
							 RL_model.predictor, 
							 n_to_generate,
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

    duration = time.time() - start
    active_threshold = 0.75

    smiles, predictions, gen_metrics = estimate_and_update(RL_model.generator, 
                                                           RL_model.predictor, 1000, 
                                                           plot=False, 
                                                           threshold=active_threshold,
                                                           return_metrics=True)
    mol_data = pd.DataFrame(dict(smiles=smiles, predictions=predictions))
    mol_actives = mol_data[mol_data.predictions > active_threshold]
    # save total data, not just actives
    if save_path:
        mol_data.to_csv(save_path, index=False, header=False)

    egfr_data = pd.read_csv('../project/datasets/egfr_with_pubchem1.csv')
    egfr_actives = egfr_data[egfr_data.predictions > active_threshold]

    metrics = compare_libraries(mol_actives, egfr_actives, properties=['MolWt', 'MolLogP'],
                                return_metrics=True, plot=False)
    results = {}
    results.update(gen_metrics)
    results.update(metrics)
    results['duration'] = duration
    # move save_path from params to results
    if save_path:
        results['save_path'] = params.pop('save_path')
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
