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

def get_fingerprint_similarities(data, baseline=None, kind='Similarity', 
                                 fingerprint=Chem.RDKFingerprint, fingerprint_params={},
                                 similarity=DataStructs.FingerprintSimilarity, sample_size=1000):
    assert kind in ('Similarity', 'MaxSimilarity')
    prop = kind
    is_internal = (baseline is None)
    if not isinstance(data, list):
        data = [data]
    n_data = len(data)
    if not is_internal:
        data0 = [baseline] * n_data
    else:
        data0 = data
    # subsample data and baseline data if present
    data_sample, data0_sample = [sample_size] * n_data, [sample_size] * n_data
    if sample_size <= 1.0:
        data_sample = [int(len(d) * sample_size) for d in data]
        data0_sample = [int(len(d0) * sample_size) for d0 in data0]
    data_sample = [min(len(d), d_sample) for d, d_sample in zip(data, data_sample)]
    data0_sample = [min(len(d0), d0_sample) for d0, d0_sample in zip(data0, data0_sample)]
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
    return sims


def get_property(data, baseline=None, prop=None):
    # if prop is a tuple, unpack. First element should be prop name, and 
    # second element should be a function on rdkit.Mol objects
    is_internal = (baseline is None)
    n_data = len(data)
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
   # results = [{f'mean_{prop}': mean,
    #           f'std_{prop}': std}
     #          for mean, std in zip(data_mean, data_std)]
   # if not is_internal:
    #    for i in range(len(results)):
     #       results[i][f'effect_{prop}'] = effect[i]
    if is_internal:
        return data
    else:
        return data, baseline


def _plot_similarities(data, baseline=None, kind='Similarity', fingerprint=Chem.RDKFingerprint, fingerprint_params={},
                       similarity=DataStructs.FingerprintSimilarity, sample_size=1000, color=None, alpha=0.3,
                       label=None, plot=True, **kwargs):
    is_internal = (baseline is None)
    prop = kind
    if not isinstance(data, list):
        data = [data]
    n_data = len(data)
    sims = get_fingerprint_similarities(data, baseline=baseline, fingerprint=fingerprint,
                                        fingerprint_params=fingerprint_params,
                                        similarity=similarity, sample_size=sample_size)
    means = [sim.mean() for sim in sims]
    stds = [sim.std() for sim in sims]
    label_kind = 'Internal' if is_internal else 'External'
    print(f'Mean {label_kind} {prop}:', '\t'.join([str(m) for m in means]))
    print(f'Std {label_kind} {prop}:', '\t'.join([str(s) for s in stds]))

    if plot:
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
            _safe_kdeplot(sim.reshape(-1), shade=True, color=c, 
                          alpha=alpha, linewidth=0, **kwargs)
            _safe_kdeplot(sim.reshape(-1), shade=False, color=c, 
                          alpha=0.3, label=lab, **kwargs)
            plt.axvline(mean, ls='-', c=c, alpha=0.3)
            plt.axvline(mean-std, ls='--', color=c, alpha=alpha)
            plt.axvline(mean+std, ls='--', color=c, alpha=alpha)

        plt.xlabel(prop)
        plt.title(f'Distribution of {prop}')
    prop = prop.lower()

    if is_internal:
        results = [{f'mean_internal_{prop}': mean,
                    f'std_internal_{prop}': std}
                   for mean, std in zip(means, stds)]
    else:
        results = [{f'mean_external_{prop}': mean,
                    f'std_external_{prop}': std}
                   for mean, std in zip(means, stds)]
    if len(results) == 1:
        results = results[0]
    return results

def _plot_property(data, baseline=None, prop=None,
                   color=None, alpha=0.3, label=None, plot=True, **kwargs):
    is_internal = (baseline is None)
    if not isinstance(data, list):
        data = [data]
    n_data = len(data)
    data = get_property(data, baseline, prop=prop)
    if not is_internal:
        data, baseline = data
    data_mean = [df.props.mean() for df in data]
    data_std = [df.props.std() for df in data]
    print(f'Mean {prop}:', '\t'.join([str(m) for m in data_mean]))
    print(f'Std {prop}:', '\t'.join([str(s) for s in data_std]))

    if not is_internal:
        baseline_mean = baseline.props.mean()
        tot_std = [pd.concat((df.props, baseline.props)).std()
                   for df in data]
        effect = (np.array(data_mean) - baseline_mean) / np.array(tot_std)
        print(f'Effect {prop}:', '\t'.join([str(e) for e in effect]))

    if plot:
        if not is_internal:
            _safe_kdeplot(baseline.props.tolist(), color='grey', shade=True, alpha=alpha, **kwargs)
            _safe_kdeplot(baseline.props, color='grey', shade=False, alpha=0.3, label=f'Baseline {prop}', **kwargs)
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
            _safe_kdeplot(df.props.tolist(), color=c, shade=True, alpha=alpha, linewidth=0, **kwargs)
            _safe_kdeplot(df.props, color=c, shade=False, alpha=0.3, label=lab, **kwargs)
            plt.axvline(mean, ls='-', color=c, alpha=0.3)
            plt.axvline(mean-std, ls='--', color=c, alpha=alpha)
            plt.axvline(mean+std, ls='--', color=c, alpha=alpha)
        plt.xlabel(prop)
        plt.title(f'Distribution of {prop}')
    results = [{f'mean_{prop}': mean,
               f'std_{prop}': std}
               for mean, std in zip(data_mean, data_std)]
    if not is_internal:
        for i in range(len(results)):
            results[i][f'effect_{prop}'] = effect[i]
    if len(results) == 1:
        results = results[0]
    return results

# now supports vectorized plotting.
def plot_property_distribution(data, baseline=None, prop=None, from_smiles=False,
                                fingerprint=Chem.RDKFingerprint, fingerprint_params={},
                                similarity=DataStructs.FingerprintSimilarity,
                                sample_size=1000, color=None, alpha=0.3, label=None, plot=True, **kwargs):
    is_vector = isinstance(data, list)
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

def get_scaffolds(data, molCol='molecules', smilesCol='smiles', aggregate='mean'):
    data = pd.DataFrame(data)
    def safeScaffoldSmiles(**kwargs):
        try:
            return MurckoScaffoldSmiles(**kwargs)
        except:
            return None
    if molCol in data.columns:
        data['scaffolds'] = data[molCol].apply(lambda m: safeScaffoldSmiles(mol=m))
    else:
        data['scaffolds'] = data[smilesCol].apply(lambda s: safeScaffoldSmiles(smiles=s))
    
    by_scaffolds = (data
                   .dropna(subset=['scaffolds'])
                   .groupby('scaffolds'))
    scaffold_data = (by_scaffolds[['scaffolds']]
                    .count()
                    .rename(lambda s: 'counts', axis=1))
    scaffold_data[molCol] = scaffold_data.index.map(Chem.MolFromSmiles)
    # aggregate numerical data by scaffold
    if len(data._get_numeric_data().columns) > 0:
        agg_data = by_scaffolds.agg(aggregate)
        agg_data.columns = agg_data.columns.map(lambda c:'_'.join(c)
                                                if isinstance(c, pd.MultiIndex)
                                                else c)
        scaffold_data = scaffold_data.join(agg_data)
    scaffold_data = (scaffold_data
                    .sort_values(by='counts', ascending=False))
    return scaffold_data


def plot_scaffolds(data, baseline=None, kind='mols',
                   n_to_show=12, molsPerRow=3,
                   color='red', alpha=0.3, 
                   label='Generated scaffold counts', 
                   bins=50, plot=True, **kwargs):
    assert kind in ('mols', 'hist')
    data = get_scaffolds(data)
    generated_scaffolds = len(data)
    if baseline is not None:
        baseline = get_scaffolds(baseline)
        # determine shared scaffolds
        shared = data.merge(baseline, how='inner', on='scaffolds')
        shared['min_counts'] = shared[['counts_x', 'counts_y']].min(axis=1)
        shared = shared.sort_values(by='min_counts', ascending=False)
        shared_scaffolds = len(shared)
        novel_scaffolds = generated_scaffolds - shared_scaffolds
        try:
            novel_fraction = novel_scaffolds / generated_scaffolds
        except ZeroDivisionError:
            novel_fraction = float('nan')
    
    if plot:
        if kind == 'hist':
            if baseline is not None:
                plt.hist(baseline.counts, color='grey', alpha=alpha, 
                        label='Baseline scaffold counts', bins=bins, **kwargs)
            plt.hist(data.counts, color=color, alpha=alpha, 
                    label=label, bins=bins, **kwargs)
            plt.xlabel('Scaffold counts')
            plt.ylabel('Number of scaffolds')
            plt.yscale('log')
            plt.title('Distribution of scaffold counts')
        else:
            if baseline is not None:
                sample = shared.iloc[:n_to_show]
                sample_smiles = list(sample.index)
                sample_labels = ['%d/%d' % (ct1, ct2) for ct1, ct2 in zip(sample.counts_y,
                                                                          sample.counts_x)]
                sample_labels[0] += ' (baseline/generated)'
            else:
                sample = data.iloc[:n_to_show]
                sample_smiles = list(sample.index)
                sample_labels = ['Counts: %d' % ct for ct in sample.counts]
            #print(sample_smiles)
            sample_mols = [Chem.MolFromSmiles(sm) for sm in sample_smiles]
            plt.imshow(Chem.Draw.MolsToGridImage(sample_mols, legends=sample_labels,
                                                molsPerRow=molsPerRow))
            if baseline is not None:
                plt.title('Shared scaffolds')
            else:
                plt.title('Generated scaffolds')

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

def compare_libraries(data, baseline, smilesCol='smiles', labelCol='predictions', molCol='molecules',
                      aggregate='mean', properties={}, sample_size=1000,
                      n_results=20, return_metrics=False, return_scaffolds=False,
                      return_novel_scaffolds=False, return_shared_scaffolds=False, bins=50, plot=True):
    if plot:
        plt.figure()
    internal_sim = _plot_similarities(data[molCol], 
                                      sample_size=sample_size, 
                                      plot=plot)
    if plot:
        plt.figure()
    external_sim = _plot_similarities(data[molCol], baseline[molCol], 
                                      sample_size=sample_size, 
                                      plot=plot)
    
    property_summary = {}
    for prop in properties:
        res = _plot_property(data[molCol], baseline[molCol], prop, plot=plot)
        property_summary.update(res)
    
    baseline_scaffolds = get_scaffolds(baseline, aggregate=aggregate,
                                       smilesCol=smilesCol, molCol=molCol)
    data_scaffolds = get_scaffolds(data, aggregate=aggregate,
                                   smilesCol=smilesCol, molCol=molCol)
    
    head = '''
    <table>
        <thead>
            <th>Baseline scaffolds</th>
            <th>Generated scaffolds</th>
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
    if plot:
 #       sns.kdeplot(baseline_scaffolds.counts, shade=True, bw=10)
        fig, axes = plt.subplots(2, 1)
        axes = axes.flatten()
        plt.sca(axes[0])
        plt.hist(baseline_scaffolds.counts, bins=bins)
        plt.xlabel('Scaffold counts')
        plt.ylabel('Number of scaffolds')
        plt.yscale('log')
        plt.title('Distribution of scaffold counts in baseline library')
        
  #      sns.kdeplot(data_scaffolds.counts, shade=True, bw=10)
        plt.sca(axes[1])
        plt.hist(data_scaffolds.counts, bins=bins)
        plt.xlabel('Scaffold counts')
        plt.ylabel('Number of scaffolds')
        plt.yscale('log')
        plt.title('Distribution of scaffold counts in generated library')
    
    shared_scaffolds = data_scaffolds.index.intersection(baseline_scaffolds.index)
    novel_scaffolds = data_scaffolds.index.difference(shared_scaffolds)
    print('Percentage of novel scaffolds: %f%% (%d / %d)' % (100*len(novel_scaffolds) / len(data_scaffolds),
                                                             len(novel_scaffolds),
                                                             len(data_scaffolds)))
    
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
        row += '<td bgcolor="white">{}</td>'.format(df.loc[shared_scaffolds][:n_results]
                                                      .style
                                                      .hide_index()
                                                      .render())
    row += '</tr>'
    
    head += row
    head += '''
    </tbody>
    </table>
    '''
    display(HTML(head))

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
        results.append(summary_results)
    if return_scaffolds:
        results.append(set(scaffolds))
    if return_novel_scaffolds:
        results.append(set(novel_scaffolds))
    if return_shared_scaffolds:
        results.append(set(shared_scaffolds))
    
    if len(results) == 1:
        return results[0]
    if len(results) > 1:
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
        self.log_results['params'] = self.log_params

    def mine_file(self, ma_window=None):
        f = open(self.filename)
        # write template for epoch data
        template = '''[\s\S]*?(?P<epoch>[0-9]+) Training on (?P<n_instances>[0-9]+) replay instances...
Setting threshold to (?P<thresholds>[0-9.e-]+)
Policy gradient...
Loss: (?P<losses>[0-9.e-]+)
Reward: (?P<rewards>[0-9.e-]+)
(Trajectories with max counts:
(?P<max_data>[\s\S]+)\n)?Mean value of predictions: (?P<actives1>[0-9.e-]+)
Proportion of valid SMILES: (?P<valids1>[0-9.e-]+)
Sample trajectories:
(?P<sample_data1>[\s\S]+)
Policy gradient replay...
Mean value of predictions: (?P<actives2>[0-9.e-]+)
Proportion of valid SMILES: (?P<valids2>[0-9.e-]+)
Sample trajectories:
(?P<sample_data2>[\s\S]+)
Fine tuning...
Mean value of predictions: (?P<actives3>[0-9.e-]+)
Proportion of valid SMILES: (?P<valids3>[0-9.e-]+)
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
                    for key in ('thresholds', 'losses', 'rewards'):
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
                        for key in ('losses', 'rewards', 'max_counts',
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

    def show_sample_mols(self, epoch, kind='fine_tune', **kwargs):
        smiles = self.sample_smiles(epoch, kind=kind)
        if not smiles:
            return
        plt.figure()
        mols = [Chem.MolFromSmiles(sm) for sm in smiles]
        im = np.array(Chem.Draw.MolsToGridImage(mols, **kwargs))
        plt.imshow(im)

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
    import os
    os.chdir('../../../ReLeaSE')
    read_path = '../project_old/benchmark/ReLeaSE/min_dist200929.log'
    read_path = '../project_old/benchmark/old_new201014.log'
    read_path = '../project_old/benchmark/ReLeaSE/n_fine_tune201031.log'
    path = '../project_old/benchmark/ReLeaSE/n_fine_tune201101'
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
    res = compare_libraries(mol_data[0], baseline=exp_data, properties=['MolWt', 'MolLogP'], return_metrics=True)
    print(res)
    plt.figure()
    plot_property_distribution(exp_data.molecules, prop='MaxSimilarity', label='Baseline',
                               alpha=alpha, sample_size=sample_size)
    plot_property_distribution(mols, prop='MaxSimilarity', baseline=exp_data.molecules,
                              label=None, alpha=alpha, sample_size=sample_size)

    fig, axes = plt.subplots(2,2)
    axes = axes.flatten()
    for ax, prop in zip(axes, ['Similarity', 'MaxSimilarity', 'MolWt', 'RingCount']):
        plt.sca(ax)
        plot_property_distribution(mols, prop=prop, baseline=exp_data.molecules, color=None,
                                   label=labs, alpha=alpha, sample_size=sample_size)
    fig, axes = plt.subplots(2,2)
    axes = axes.flatten()
    for i, data in enumerate(mol_data):
        plt.sca(axes[i])
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
