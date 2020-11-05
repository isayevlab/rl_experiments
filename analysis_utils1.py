# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 5:50:13 2020

Newer implementation of analysis_utils functionalities.
Merge pairwise fingerprint similarities with property comparisons.
Implementations will be friendlier to embedding in matplotlib.pyplot applications.
This should streamline overlay plots.
Also eliminate functions that appear superfluous.

@author: niles
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import ast
from collections import defaultdict

import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

def _safe_kdeplot(*args, **kwargs):
    try:
        sns.kdeplot(*args, **kwargs)
    except:
        sns.kdeplot(*args, bw=0.2, **kwargs)

def _plot_similarities(data, baseline=None, kind='Similarity', fingerprint=Chem.RDKFingerprint, fingerprint_params={},
                       similarity=DataStructs.FingerprintSimilarity, sample_size=1000, color=None, alpha=0.3,
                       label=None, plot=True, **kwargs):
    assert kind in ('Similarity', 'MaxSimilarity')
    prop = kind
    is_internal = (baseline is None)
    n_data = len(data)
    if not is_internal:
        data0 = [baseline] * n_data
    else:
        data0 = data
    data_sample, data0_sample = [sample_size] * n_data, [sample_size] * n_data
    if sample_size <= 1.0:
        data_sample = [int(len(d) * sample_size) for d in data]
        data0_sample = [int(len(d0) * sample_size) for d0 in data0]
    data_sample = [min(len(d), d_sample) for d, d_sample in zip(data, data_sample)]
    data0_sample = [min(len(d0), d0_sample) for d0, d0_sample in zip(data0, data0_sample)]
    # subsample data and baseline data if present
    data = [np.random.choice(d, d_sample, replace=False) for d, d_sample in zip(data, data_sample)]
    if is_internal:
        data0 = data
    else:
        tmp = np.random.choice(baseline, data0_sample[0], replace=False)
        data0 = [tmp] * n_data
    
    data_fps = [[fingerprint(mol, **fingerprint_params) for mol in d] for d in data]
    if is_internal:
        data0_fps = data_fps
    else:
        # save some overhead by reusing same reference
        tmp = [fingerprint(mol, **fingerprint_params) for mol in data0[0]]
        data0_fps = [tmp] * n_data
    
    sims = []
    for d_fps, d0_fps in zip(data_fps, data0_fps):
        sim = np.array([[similarity(x_fp, y_fp) for x_fp in d_fps] for y_fp in d0_fps])
        if kind == 'MaxSimilarity':
            # exclude trivial self-similarity for internal comparison
            if is_internal:
                cond = lambda x: x < 1
            else:
                cond = lambda x: True
            sim = [max(filter(cond, l)) for l in sim]
            sim = np.array(sim)
        sims.append(sim)

    means = [sim.reshape(-1).mean() for sim in sims]
    stds = [sim.reshape(-1).std() for sim in sims]
    if plot:
        old_ax = plt.gca()
        # direct all drawing to new ax if specified
        if kwargs.get('ax'):
            plt.sca(kwargs['ax'])
        if label is None:
            label_kind = 'Internal' if is_internal else 'External'
            if n_data == 1:
                label = [f'{label_kind} {prop}']
            else:
                label = [f'{label_kind} {prop} %d' % (i+1) for i in range(n_data)]
        if color is None:
            if n_data == 1:
                color = ['red']
            else:
                color = ['C%d' % i for i in range(n_data)]
        for i in range(n_data):
            sim = sims[i]
            mean = means[i]
            std = stds[i]
            c = color[i]
            lab = label[i]
            _safe_kdeplot(sim.reshape(-1), shade=True, color=c, alpha=alpha,
                        label=lab, **kwargs)
            plt.axvline(mean, ls='-', c=c, alpha=0.3)
            plt.axvline(mean-std, ls='--', c=c, alpha=0.3)
            plt.axvline(mean+std, ls='--', c=c, alpha=0.3)

        plt.xlabel(prop)
        plt.title(f'Distribution of {prop}')
        plt.sca(old_ax)
    
    if is_internal:
        results = [{f'mean_internal_{prop}': mean,
                    f'std_internal_{prop}': std}
                   for mean, std in zip(means, stds)]
    else:
        results = [{f'mean_external_{prop}': mean,
                    f'std_external_{prop}': std}
                   for mean, std in zip(means, stds)]
    return results

# support vectorized plotting (list of data, optional baseline (not vector)
def _plot_property(data, baseline=None, prop=None,
                   color=None, alpha=0.3, label=None, plot=True, **kwargs):
    # if prop is a tuple, unpack. First element should be prop name, and 
    # second element should be a function on rdkit.Mol objects
    is_internal = (baseline is None)
    n_data = len(data)
    if isinstance(prop, tuple):
        assert len(prop) == 2
        assert isinstance(prop[0], str)
        prop, prop_func = prop
    else:
        prop_func = getattr(Descriptors, prop)
    def get_prop(mol):
        try:
            return prop_func(mol)
        except:
            return None
    tmp = []
    for d in data:
        df = pd.DataFrame(dict(molecules=d))
        df['props'] = df.molecules.apply(get_prop)
        df = df.dropna()
        tmp.append(df)
    data = tmp
    if not is_internal:
        baseline = pd.DataFrame(dict(molecules=baseline))
        baseline['props'] = baseline.molecules.apply(get_prop)
        baseline = baseline.dropna()
    data_mean = [df.props.mean() for df in data]
    data_std = [df.props.std() for df in data]
    if not is_internal:
        baseline_mean = baseline.props.mean()
        tot_std = [pd.concat((df.props, baseline.props)).std()
                   for df in data]
        effect = (np.array(data_mean) - baseline_mean) / np.array(tot_std)
    if plot:
        old_ax = plt.gca()
        if kwargs.get('ax'):
            plt.sca(kwargs['ax'])
        if not is_internal:
            _safe_kdeplot(baseline.props, color='grey', shade=True, alpha=alpha, label=f'Baseline {prop}')
            plt.axvline(baseline_mean, ls='-', c='k', alpha=0.3)
        if label is None:
            if n_data == 1:
                label = [f'Generated {prop}']
            else:
                label = [f'Generated {prop} %d' % (i+1) for i in range(n_data)]
        if color is None:
            if n_data == 1:
                color = ['red']
            else:
                color = ['C%d' % i for i in range(n_data)]
        for i in range(len(data)):
            df = data[i]
            c = color[i]
            lab = label[i]
            mean, std = data_mean[i], data_std[i]
            _safe_kdeplot(df.props, color=c, shade=True, alpha=alpha, label=lab)
            plt.axvline(mean, ls='-', color=c, alpha=0.3)
            plt.axvline(mean-std, ls='--', color=c, alpha=0.3)
            plt.axvline(mean+std, ls='--', color=c, alpha=0.3)
        plt.xlabel(prop)
        plt.title(f'Distribution of {prop}')
        plt.sca(old_ax)
    results = [{f'mean_{prop}': mean,
               f'std_{prop}': std}
               for mean, std in zip(data_mean, data_std)]
    if not is_internal:
        for i in range(len(results)):
            results[i][f'effect_{prop}'] = effect[i]
    
    return results

# now supports vectorized plotting.
def plot_property_distribution(data, baseline=None, prop=None, from_smiles=False,
                                fingerprint=Chem.RDKFingerprint, fingerprint_params={},
                                similarity=DataStructs.FingerprintSimilarity,
                                sample_size=1000, color=None, alpha=0.3, label=None, plot=True, **kwargs):
    is_vector = True
    data = list(data)
    if len(data) > 0:
        tmp = data[0]
        # test to see depth of data
        if isinstance(tmp, str) or isinstance(tmp, rdkit.Chem.Mol):
            is_vector = False
    # if not vector, wrap all necessary attributes in list
    if not is_vector:
        data = [data]
        color = [color] if color else None
        label = [label] if label else None
    else:
        # make sure other parameters are vectorized
        if label is not None:
            assert len(label) == len(data)
        if color is not None:
            assert len(color) == len(data)
    if from_smiles:
        data = [[Chem.MolFromSmiles(sm) for sm in d] for d in data]
        if baseline is not None:
            baseline = [Chem.MolFromSmiles(sm) for sm in baseline]
    # compute pairwise fingerprint similarities and plot
    if prop in ('Similarity', 'MaxSimilarity'):
        results = _plot_similarities(data, baseline=baseline, kind=prop, fingerprint=fingerprint,
                                     fingerprint_params=fingerprint_params, similarity=similarity,
                                     sample_size=sample_size, color=color, alpha=alpha, 
                                     label=label, plot=plot, **kwargs)
    else:
        results = _plot_property(data, baseline=baseline, prop=prop, color=color, 
                                 alpha=alpha, label=label, plot=plot, **kwargs)
    if len(results) == 1:
        results = results[0]
    return results

def plot_scaffolds(data, baseline=None, kind='mols', from_smiles=False,
                   n_to_show=12, molsPerRow=3,
                   color='red', alpha=0.3, label='Generated scaffold counts', bins=50, **kwargs):
    assert kind in ('mols', 'hist')

    # convert to smiles to use MurckoScaffoldSmiles function
    def safe_scaffold_smiles(smiles=None, mol=None):
        try:
            scaffold = MurckoScaffoldSmiles(smiles=smiles, mol=mol)
        except:
            scaffold = None
        return scaffold

    if not from_smiles:
        get_scaffold_smiles = lambda mol: safe_scaffold_smiles(mol=mol)
    else:
        get_scaffold_smiles = safe_scaffold_smiles

    data = pd.DataFrame(dict(data_=data))
    data['scaffold_smiles'] = data.data_.apply(get_scaffold_smiles)
    data = data.dropna()
    data = data.groupby('scaffold_smiles').count()
    data['counts'] = data.data_
    generated_scaffolds = len(data)
    if baseline is not None:
        baseline = pd.DataFrame(dict(data_=baseline))
        baseline['scaffold_smiles'] = baseline.data_.apply(get_scaffold_smiles)
        baseline.dropna()
        baseline = baseline.groupby('scaffold_smiles').count()
        
        # determine shared scaffolds
        data['counts1'] = data.counts
        baseline['counts'] = baseline.data_
        baseline['counts0'] = baseline.counts
        shared = pd.concat((data, baseline), axis=1)
        shared = shared.dropna()
        shared_scaffolds = len(shared)
        novel_scaffolds = generated_scaffolds - shared_scaffolds
        try:
            novel_fraction = novel_scaffolds / generated_scaffolds
        except ZeroDivisionError:
            novel_fraction = float('nan')

    if kind == 'hist':
        ax = plt.gca()
        ax = kwargs.get('ax', ax)
        if baseline is not None:
            ax.hist(baseline.counts, color='grey', alpha=alpha, 
                    label='Baseline scaffold counts', bins=bins, **kwargs)
        ax.hist(data.counts, color=color, alpha=alpha, 
                label=label, bins=bins, **kwargs)
        ax.set_xlabel('Scaffold counts')
        ax.set_ylabel('Number of scaffolds')
        ax.set_yscale('log')
        ax.set_title('Distribution of scaffold counts')
    else:
        if baseline is not None:
            shared['min_counts'] = [min(ct1, ct2) for ct1, ct2 in zip(shared.counts1, shared.counts0)]
            shared = shared.sort_values(by='min_counts', ascending=False)
            sample = shared.iloc[:n_to_show]
            sample_smiles = list(sample.index)
            sample_labels = ['%d/%d' % (ct1, ct2) for ct1, ct2 in zip(sample.counts0,
                                                                      sample.counts1)]
            sample_labels[0] += ' (baseline/generated)'
        else:
            data = data.sort_values(by='counts', ascending=False)
            sample = data.iloc[:n_to_show]
            sample_smiles = list(sample.index)
            sample_labels = ['Counts: %d' % ct for ct in sample.counts]
        sample_mols = [Chem.MolFromSmiles(sm) for sm in sample_smiles]
        ax = plt.gca()
        ax = kwargs.get('ax', ax)
        ax.imshow(Chem.Draw.MolsToGridImage(sample_mols, legends=sample_labels,
                                            molsPerRow=molsPerRow))
        if baseline is not None:
            ax.set_title('Shared scaffolds')
        else:
            ax.set_title('Generated scaffolds')
    results = {'generated_scaffolds': generated_scaffolds}
    if baseline is not None:
        results['novel_scaffolds'] = novel_scaffolds
        results['novel_fraction'] = novel_fraction
    return results

from rdkit.Chem import Descriptors

# TO DO: 
# v add options to return various parts
# - add option to aggregate more columns (currently only label)
# - option to display number of hits
# - see if can display parallel columns
from IPython.display import display, HTML
from rdkit.Chem import PandasTools

def compare_libraries(data, baseline, smilesCol='smiles', labelCol='predictions', molCol=None,
                      aggregate='mean', properties={}, sample_size=1000,
                      n_results=20, return_metrics=False, return_scaffolds=False,
                      return_novel_scaffolds=False, return_shared_scaffolds=False, bins=50, plot=True, show=True, **kwargs):
    if not plot:
        show = False
    if plot:
        plt.figure()
    internal_sim = pairwise_fingerprint_similarities(data[smilesCol], sample_size=sample_size, 
                                                     plot=plot, show=False, **kwargs)
    if plot:
        plt.figure()
    external_sim = pairwise_fingerprint_similarities(data[smilesCol], baseline[smilesCol], 
                                                     sample_size=sample_size, 
                                                     plot=plot, show=False, **kwargs)
    
    property_summary = compare_libraries_by_properties(data, baseline, properties, 
                                                       plot=plot, show=False, **kwargs)
    
    baseline_scaffolds = get_scaffolds(baseline, smilesCol=smilesCol, molCol=molCol, 
                                       columns=[labelCol], aggregate=aggregate)
    data_scaffolds = get_scaffolds(data, smilesCol=smilesCol, molCol=molCol,
                                   columns=[labelCol], aggregate=aggregate)
    
    head = '''
    <table>
        <thead>
            <th> Baseline scaffolds</th>
            <th> Generated scaffolds</th>
        </thead>
        <tbody>
    '''
    
    row = '<tr>'
    
    for df in (baseline_scaffolds, data_scaffolds):
        row += '<td bgcolor="white">{}</td>'.format(df.sort_values(by='counts', ascending=False)[:n_results].\
                                                    style.hide_index().render())
    row += '</tr>'
    
    head += row
    head += '''
    </tbody>
    </table>
    '''
    
    display(HTML(head))
    scaffolds = (data_scaffolds, baseline_scaffolds)
    '''
    print('\nBaseline scaffolds:')
    display(baseline_scaffolds.sort_values(by='counts', ascending=False)[:n_results])
    data_scaffolds = get_scaffolds(data, smilesCol=smilesCol, molCol=molCol, 
                                   columns=[labelCol], aggregate=aggregate)
    print('\nGenerated scaffolds:')
    display(data_scaffolds.sort_values(by='counts', ascending=False)[:n_results])
    scaffolds = (data_scaffolds, baseline_scaffolds)
    '''
    if plot:
  #      plt.figure()
 #       sns.kdeplot(baseline_scaffolds.counts, shade=True, bw=10)
        baseline_scaffolds.hist('counts', bins=bins)
        plt.title('Distribution of scaffold counts in baseline library')
        plt.xlabel('Scaffold counts')
        plt.ylabel('Number of scaffolds')
        plt.gca().set_yscale('log')
        
   #     plt.figure()
  #      sns.kdeplot(data_scaffolds.counts, shade=True, bw=10)
        data_scaffolds.hist('counts', bins=bins)
        plt.title('Distribution of scaffold counts in generated library')
        plt.xlabel('Scaffold counts')
        plt.ylabel('Number of scaffolds')
        plt.gca().set_yscale('log')
    
    novel_scaffolds, shared_scaffolds = compare_scaffolds(data_scaffolds.index, baseline_scaffolds.index)
    
    head = '''
    <table>
        <thead>
            <th> Title 1</th>
            <th> Title 2</th>
        </thead>
        <tbody>
    '''
    
    row = '<tr>'
    
    for df in (baseline_scaffolds, data_scaffolds):
        row += '<td bgcolor="white">{}</td>'.format(df.loc[shared_scaffolds][:n_results].style.hide_index().render())
    row += '</tr>'
    
    head += row
    head += '''
    </tbody>
    </table>
    '''
    display(HTML(head))
    if show:
        plt.show()

    summary_results = {}
    
    summary_results.update(internal_sim)
    summary_results.update(external_sim)
    summary_results.update(property_summary)
    summary_results['generated_scaffolds'] = len(data_scaffolds)
    summary_results['novel_scaffolds'] = len(novel_scaffolds)
    try:
        novel_fraction = len(novel_scaffolds) / len(data_scaffolds)
    except ZeroDivisionError:
        novel_fraction = float('nan')
    summary_results['novel_fraction'] = novel_fraction
    
    results = []
    
    if return_metrics:
        if return_metrics is True or 'all':
            metrics = list(summary_results.keys())
        else: 
            metrics = return_metrics
        results.append({metric: summary_results[metric]
                        for metric in metrics})
    if return_scaffolds:
        results.append(scaffolds)
    if return_novel_scaffolds:
        results.append(novel_scaffolds)
    if return_shared_scaffolds:
        results.append(shared_scaffolds)
    
    if len(results) == 1:
        return results[0]
    if len(results) > 1:
        return results


# for live-time analysis on commandline
def show_mols(s, sep='\*?\n[0-9., ]*\t?', from_list=False, from_mols=False, show=True, **kwargs):
    if not from_list:
        regex = re.compile(sep)
        from_mols = False
        smiles = regex.split(s)
    else:
        smiles = list(s)
    if not from_mols:
        mols = [Chem.MolFromSmiles(sm) for sm in smiles]
    else:
        mols = list(s)
    im = np.array(Chem.Draw.MolsToGridImage(mols, **kwargs))
    plt.imshow(im)
    if show:
        plt.show()

# function to mine logs. Takes filename as input, then mines for various properties.
# may want to have object-based implementation later on.
# also may want to provide flexibility for different kinds of logs.

def mine_log(filename):
    f = open(filename)
    regex_pred = re.compile('(?<=: )[0-9.]+')
    regex_valid = regex_pred
    regex_n_instances = re.compile('(?<=Fine-tuning on )[0-9]+')
    regex_thresh = re.compile('[0-9.]+')
    regex_max_counts = re.compile('[0-9]+')
    regex_n_clusters = re.compile('[0-9]+(?= clusters)')
    results = []
    s = f.readline()

    while s != '':
        thresholds = []
        n_instances = []
        max_counts = []
        actives = []
        valids = []
        n_clusters = []
        i = 0
        finished = False
        # this implementation a bit bulky. Think of adding all instances to buffer then append at once.
        while not finished:
            s = f.readline()
            if s == '':
                break
            s = s.strip()
            max_count = 1
            if s.startswith('Setting replay threshold'):
                val = regex_thresh.search(s).group(0)
                val = float(val)
                thresholds.append(val)
            if s.endswith('instances...'):
                val = regex_n_instances.search(s).group(0)
                val = int(val)
                n_instances.append(val)
            if s.endswith('max counts:'):
                s = f.readline()
                val = regex_max_counts.search(s).group(0)
                max_count = int(val)
            if s.startswith('Percentage of predictions'):
                val = regex_pred.search(s).group(0)
                val = float(val)
                actives.append(val)
            if s.startswith('Proportion of valid SMILES:'):
                val = regex_valid.search(s).group(0)
                val = float(val)
                valids.append(val)
            if s.endswith('clusters'):
                val = regex_n_clusters.search(s).group(0)
                val = int(val)
                n_clusters.append(val)
                # may append max counts now, have reached 'end' of parsing
                max_counts.append(max_count)
            if s.startswith('Metrics'):
                train_len = len(max_counts)  # lazy, should really be min of all fields. 
                print('train len', train_len)
                thresholds = thresholds[:train_len]
                n_instances = n_instances[:train_len]
                max_counts = max_counts[:train_len]
                actives = actives[:train_len]
                valids = valids[:train_len]

                df = pd.DataFrame({'thresholds': thresholds,
                                   'n_instances': n_instances,
                                   'max_counts': max_counts,
                                   'active_fraction': actives,
                                   'valid_fraction': valids,
                                   'n_clusters': n_clusters})
                results.append(df)
                print('reached end of trace, break')
                finished = True
    return results

# TO DO:
# possibly rename LogMiner to LogData, more consistent.
# get methods of RunData, all of which to LogMiner, but passing additional required run kwarg
# possibly iterator methods for sample smiles. Must manage possible empty lists
# parse dictionaries for params and metrics, add to RunData
# possible finessing of plot, running average?
# later, can get train_rl_model to print losses and rewards. Unclear if necessary though
class LogMiner(object):
    def __init__(self, filename, ma_window=None):
        super(LogMiner, self).__init__()
        self.filename = filename
        log_data, log_params, log_metrics = self.mine_file(ma_window=ma_window)
        self.log_data = log_data
        self.log_params = log_params
        self.log_metrics = log_metrics
        self.n_runs = len(self.log_data)
        self.log_results = pd.DataFrame(log_metrics)

    def mine_file(self, ma_window=None):
        f = open(self.filename)
        # write template for epoch data
        template = '''[\s\S]*?(?P<epoch>[0-9]+) Training on (?P<n_instances>[0-9]+) replay instances...
Setting threshold to (?P<thresholds>[0-9.]+)
Policy gradient...
Mean activity in replay buffer: (?P<actives_buffer>[0-9.]+)
Loss: (?P<losses>[0-9.]+)
Reward: (?P<rewards>[0-9.]+)
(Trajectories with max counts:
(?P<max_data>[\s\S]+)\n)?Mean value of predictions: (?P<actives1>[0-9.]+)
Proportion of valid SMILES: (?P<valids1>[0-9.]+)
Sample trajectories:
(?P<sample_data1>[\s\S]+)
Policy gradient replay...
Mean value of predictions: (?P<actives2>[0-9.]+)
Proportion of valid SMILES: (?P<valids2>[0-9.]+)
Sample trajectories:
(?P<sample_data2>[\s\S]+)
Fine tuning...
Mean value of predictions: (?P<actives3>[0-9.]+)
Proportion of valid SMILES: (?P<valids3>[0-9.]+)
Sample trajectories:
(?P<sample_data3>[\s\S]+)

'''
        
        epoch_regex = re.compile(template)
        template = '''[\s\S]*Metrics for (?P<params>.+?):
(?P<metrics>.+)'''
        metric_regex = re.compile(template)
        log_data = []
        log_params = []
        log_metrics = []
        line = f.readline()
        s = line
        while line:
            finished = False
            run_data = defaultdict(list)
            while not finished:
                # if present data matches epoch template, dump data into run_data
                if epoch_regex.match(s):
                    epoch_data = {}
                    results = epoch_regex.match(s)
                    results = results.groupdict()
                    for key in ('epoch', 'n_instances'):
                        epoch_data[key] = int(results[key])
                    for key in ('thresholds', 'losses', 'rewards', 'actives_buffer'):
                        epoch_data[key] = float(results[key])
                    for i, field in enumerate(['policy', 'replay', 'fine_tune']):
                        epoch_data['actives_%s' % field] = float(results['actives%d' % (i+1)])
                        epoch_data['valids_%s' % field] = float(results['valids%d' % (i+1)])
                        epoch_data['smiles_%s' % field] = results['sample_data%d' % (i+1)].split('\n')
                    
                    max_counts = 1
                    max_samples = []
                    tmp = results['max_data']
                    if tmp is not None:
                        max_counts = int(tmp.split()[0])
                        max_samples = tmp.split()[1::2]
                    epoch_data['max_counts'] = max_counts
                    epoch_data['smiles_max'] = max_samples
                    
                    # question: which of 3 sets to include?
                    for k, v in epoch_data.items():
                        run_data[k].append(v)
                    s = ''
                elif metric_regex.match(s):
                    s = re.sub('nan', 'None', s)
                    results = metric_regex.match(s)
                    params = ast.literal_eval(results.group('params'))
                    metrics = ast.literal_eval(results.group('metrics'))
                    metrics = {k: v if v is not None else float('nan') for k, v in metrics.items()}
                    log_params.append(params)
                    log_metrics.append(metrics)
                    if ma_window:
                        for key in ('losses', 'rewards', 'actives_buffer',
                                    'actives_policy', 'valids_policy',
                                    'actives_replay', 'valids_replay',
                                    'actives_fine_tune', 'valids_fine_tune'):
                            run_data[key] = moving_average(run_data[key], n=ma_window)

                    log_data.append(RunData(run_data, params=params, metrics=metrics))
                    finished = True
                    print('reached end of trace for run %d' % len(log_data))
                    s = ''
                line = f.readline()
                if line == '':
                    break
                s += line
        print('finished mining log')
        return log_data, log_params, log_metrics
    
    # get RunData number i (starting from 1)
    def run(self, i):
        return self.log_data[i-1]

    def runs(self):
        return iter(self.log_data)
    
    def results(self):
        return self.log_results

    def __len__(self):
        return self.n_runs

    def __repr__(self):
        return '%s(filename=%r)' % (self.__class__.__name__, self.filename)
    
class RunData(object):
    def __init__(self, data, params=None, metrics=None, nan_value=None, **kwargs):
        super(RunData, self).__init__()
        data = pd.DataFrame(data, **kwargs)
        self.data = data
        self.params = params
        self.metrics = metrics

    def get_metrics(self):
        return self.metrics

    def get_params(self):
        return self.params

    def sample_smiles(self, epoch, kind='fine_tune'):
        assert kind in ('policy', 'replay', 'fine_tune', 'max')
        return list(self.loc[epoch]['smiles_%s' % kind])
    
    def smiles(self, kind='fine_tune'):
        assert kind in ('policy', 'replay', 'fine_tune', 'max')
        return self['smiles_%s' % key]

    def show_sample_mols(self, epoch, kind='fine_tune', show=True, **kwargs):
        smiles = self.sample_smiles(epoch, kind=kind)
        if not smiles:
            return
        plt.figure()
        mols = [Chem.MolFromSmiles(sm) for sm in smiles]
        im = np.array(Chem.Draw.MolsToGridImage(mols))
        plt.imshow(im)
        if show:
            plt.show()

    def __getitem__(self, items):
        return self.data.__getitem__(items)
    
    def __setitem__(self, items, values):
        self.data.__setitem__(items, values)

    def __repr__(self):
        s = '''%s(data=\n%s,\nparams=%s,\nmetrics=%s)''' % (self.__class__.__name__, 
                                                          self.data, self.params, self.metrics)
        return s
    
    def __str__(self):
        return self.data.__str__()

    def __getattr__(self, attr):
        if attr in self.__dir__():
            return getattr(super(RunData, self), attr)
        return getattr(self.data, attr)
    
    def __len__(self):
        return self.data.__len__()

def mine_file(filename, ma_window=None):
    f = open(filename)
    # write template for epoch data
    template = '''[\s\S]*?(?P<epoch>[0-9]+) Training on (?P<n_instances>[0-9]+) replay instances...
Setting threshold to (?P<thresholds>[0-9.]+)
Policy gradient...
Loss: (?P<losses>[0-9.]+)
Reward: (?P<rewards>[0-9.]+)
(Trajectories with max counts:
(?P<max_data>[\s\S]+))?
Mean value of predictions: (?P<actives1>[0-9.]+|nan)
Proportion of valid SMILES: (?P<valids1>[0-9.]+)
Sample trajectories:
(?P<sample_data1>[\s\S]+)
Policy gradient replay...
Mean value of predictions: (?P<actives2>[0-9.]+|nan)
Proportion of valid SMILES: (?P<valids2>[0-9.]+)
Sample trajectories:
(?P<sample_data2>[\s\S]+)
Fine tuning...
Mean value of predictions: (?P<actives3>[0-9.]+|nan)
Proportion of valid SMILES: (?P<valids3>[0-9.]+)
Sample trajectories:
(?P<sample_data3>[\s\S]+)

'''

    epoch_regex = re.compile(template)
    template = '''[\s\S]*Metrics for (?P<params>.+?):
(?P<metrics>.+)'''
    metric_regex = re.compile(template)
    log_data = []
    log_params = []
    log_metrics = []
    line = f.readline()
    s = line
    while s:
        finished = False
        run_data = defaultdict(list)
        while not finished:
            # if present data matches epoch template, dump data into run_data
            if epoch_regex.match(s):
                epoch_data = {}
                results = epoch_regex.match(s)
                results = results.groupdict()
                for key in ('epoch', 'n_instances'):
                    epoch_data[key] = int(results[key])
                for key in ('thresholds', 'losses', 'rewards'):
                    epoch_data[key] = float(results[key])
                for i, field in enumerate(['policy', 'replay', 'fine_tune']):
                    epoch_data['actives_%s' % field] = float(results['actives%d' % (i+1)])
                    epoch_data['valids_%s' % field] = float(results['valids%d' % (i+1)])
                    epoch_data['smiles_%s' % field] = results['sample_data%d' % (i+1)].split('\n')
                
                max_counts = 1
                max_samples = []
                tmp = results['max_data']
                if tmp != '':
                    max_counts = int(tmp.split()[0])
                    max_samples = tmp.split()[1::2]
                epoch_data['max_counts'] = max_counts
                epoch_data['smiles_max'] = max_samples
                
                # question: which of 3 sets to include?
                for k, v in epoch_data.items():
                    run_data[k].append(v)
                s = ''
            elif metric_regex.match(s):
                s = re.sub('nan', 'None', s, flags=re.MULTILINE)
                results = metric_regex.match(s)
                params = ast.literal_eval(results.group('params'))
                metrics = ast.literal_eval(results.group('metrics'))
                metrics = {k: v if v is not None else float('nan') for k, v in metrics.items()}
                log_params.append(params)
                log_metrics.append(metrics)
                log_data.append(RunData(run_data, params=params, metrics=metrics))
                finished = True
                print('reached end of trace for run %d' % len(log_data))
                s = ''
            s += f.readline()
            if s == '':
                break
    print('finished mining log')
    return log_data, log_params, log_metrics

def moving_average(l, n=10):
    l = np.array(l)
    ma = np.cumsum(l, dtype=float)
    middle = ma[n:] - ma[:-n]
    middle = middle / n
    head = ma[:n] / np.arange(1,n+1)
    ma[:n] = head
    ma[n:] = middle
    return ma

if __name__ == '__main__':
    from pandas.plotting import scatter_matrix
    read_path = '../project/benchmark/ReLeaSE/min_dist200929.log'
    read_path = '../project/benchmark/old_new201014.log'
    read_path = '../project/benchmark/ReLeaSE/n_fine_tune201031.log'
    a,b,c = (0,0,0)#mine_file(read_path)
    print(a)
    print(b)
    print(c)
    if False:
        log_miner = LogMiner(read_path, ma_window=10)
        print('LogMiner name: %s' % log_miner)
        print('displaying runs...')
        n_rows = len(log_miner.run(1)._get_numeric_data().columns)
        fig, axes = plt.subplots(n_rows, n_rows)
        for run in log_miner.runs():
            scatter_matrix(run, ax=axes)
        plt.show()
        print('%r' % run)
        print(log_miner.log_data[0].columns)
        print(log_miner.log_results)
        run = log_miner.run(1)
        print(run.params)
     #   for i in run.index[::50]:
      #      run.show_sample_mols(i, kind='cluster', show=False)
      #  plt.show()
       # print(run.metrics)
        print('mining log...')
       # results = mine_log(read_path)
       # print(results)
        
        # test out some plotting capabilities
    if True:
        path = '../project/benchmark/ReLeaSE/n_fine_tune201101'
        alpha = 0.3
        sample_size = 1000
        mol_data = []
        for i in range(4):
            lib_path = path + '-%d.smi' % (i+1) #(2*i+1)
            tmp = pd.read_csv(lib_path, names=['smiles','predictions'])
            tmp = tmp.copy()[tmp.predictions > 0.75]
            tmp['molecules'] = tmp.smiles.apply(Chem.MolFromSmiles)
            mol_data.append(tmp)
        #mol_data = mol_data[3:6]
        exp_data = pd.read_csv('../project/datasets/egfr_unbiased.smi', names=['smiles','predictions'])#, 'predictions'])
        exp_data['molecules'] = exp_data.smiles.apply(Chem.MolFromSmiles)
        labs = ['run %d' % (i+1) for i in range(len(mol_data))]
        mols = [df.molecules for df in mol_data]
        plt.figure()
        plot_property_distribution(exp_data.molecules, prop='MaxSimilarity', label='Baseline',
                                   alpha=alpha, sample_size=sample_size)
        plot_property_distribution(mols, prop='MaxSimilarity', baseline=exp_data.molecules,
                                  label=None, alpha=alpha, sample_size=sample_size)

        fig, axes = plt.subplots(2,2)
        axes = axes.flatten()
        for ax, prop in zip(axes, ['Similarity', 'MaxSimilarity', 'MolWt', 'RingCount']):
            plot_property_distribution(mols, prop=prop, baseline=exp_data.molecules, ax=ax, color=None,
                                       label=labs, alpha=alpha, sample_size=sample_size)
        fig, axes = plt.subplots(2,2)
        axes = axes.flatten()
        for i, data in enumerate(mol_data):
            plot_scaffolds(data.molecules, kind='mols', ax=axes[i])
            axes[i].set_title('Generated %d' % (i+1))
        #plt.sca(axes[2])
        #plot_scaffolds(mol_data[0].smiles, kind='mols', baseline=exp_data.smiles, from_smiles=True)
        plt.figure()
        plot_scaffolds(mol_data[0].molecules, kind='hist', baseline=exp_data.molecules, label='run 1', alpha=alpha)
        for i, data in enumerate(mol_data[1:]):
            plot_scaffolds(data.molecules, kind='hist', label='run %d' % (i+2), color='C%d' % (i+2), alpha=alpha)
        plt.legend()
        plt.show()

    if False: 
        mol_data = pd.read_csv('../project/datasets/cdk1_clf_augmented_ReLeaSE(2).csv')
        mol_data = pd.read_csv('../project/benchmark/ReLeaSE/model_memory201019-1.smi', names=['smiles','predictions'])
        mol_data1 = pd.read_csv('../project/benchmark/ReLeaSE/replay_combo201023-1.smi', names=['smiles','predictions'])
        exp_data = pd.read_csv('../project/datasets/egfr_actives.smi')#, 'predictions'])
        #exp_actives = exp_data.copy()[exp_data.predictions > 0.75]
        #compare_libraries(mol_data, exp_data, properties=['MolWt', 'MolLogP'])
        mol_data['molecules'] = mol_data.smiles.apply(Chem.MolFromSmiles)
        mol_data1['molecules'] = mol_data1.smiles.apply(Chem.MolFromSmiles)
        exp_data['molecules'] = exp_data.smiles.apply(Chem.MolFromSmiles)
        fig, axes = plt.subplots(2,2)
        axes = axes.flatten()
        for ax, prop in zip(axes, ['Similarity', 'MaxSimilarity', 'MolWt', 'RingCount']):
            plot_property_distribution(mol_data.molecules, prop, baseline=exp_data.molecules, ax=ax, from_smiles=False)
            plot_property_distribution(mol_data1.molecules, prop, ax=ax, color='blue', from_smiles=False, label='lib1')
        fig, axes = plt.subplots(1,3)
        plt.sca(axes[0])
        plot_scaffolds(exp_data.molecules, kind='mols')
        plt.title('Baseline scaffolds')
        plt.sca(axes[1])
        plot_scaffolds(mol_data.molecules, kind='mols')
        plt.title('Generated scaffolds')
        plt.sca(axes[2])
        plot_scaffolds(mol_data.smiles, kind='mols', baseline=exp_data.smiles, from_smiles=True)
        plt.figure()
        plot_scaffolds(mol_data.molecules, kind='hist', baseline=exp_data.molecules)
        plot_scaffolds(mol_data1.molecules, kind='hist', label='lib1', color='blue')
        plt.show()

