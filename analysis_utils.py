# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 15:50:13 2020

@author: niles_x0odhz5
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import ast
from collections import defaultdict

from rdkit import Chem, DataStructs
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

# NOTE: consider removing mol feature, move to different method. Can remove sorting functions
def pairwise_fingerprint_similarities(x, y=None, fingerprint=Chem.RDKFingerprint, fingerprint_params={},
                                      similarity=DataStructs.FingerprintSimilarity,
                                      from_smiles=True, by_scaffold=False, scaffold_threshold=1, sample_size=1000,
                                      plot_hist=True, plot=True, show=True, color='red', alpha=0.3, **kwargs):
    
    is_internal = False
    if y is None:
        is_internal = True
        y = x
    if not plot:
        show = False
    if from_smiles:
        x = [Chem.MolFromSmiles(sm) for sm in x]
        y = [Chem.MolFromSmiles(sm) for sm in y]
    
    if by_scaffold:
        x = [MurckoScaffoldSmiles(mol=m) for m in x]
        y = [MurckoScaffoldSmiles(mol=m) for m in y]
        x_unique_scaffolds, x_cts = np.unique(x, return_counts=True)
        y_unique_scaffolds, y_cts = np.unique(y, return_counts=True)
        x = x_unique_scaffolds[x_cts >= scaffold_threshold]
        y = y_unique_scaffolds[y_cts >= scaffold_threshold]
        x = [Chem.MolFromSmiles(sm) for sm in x]
        y = [Chem.MolFromSmiles(sm) for sm in y]
    
    x_sample, y_sample = sample_size, sample_size
    if sample_size <= 1.0:
        x_sample = int(len(x) * sample_size)
        y_sample = int(len(y) * sample_size)
    x_sample = min(len(x), x_sample)
    y_sample = min(len(y), y_sample)
    x = np.random.choice(x, x_sample, replace=False)
    y = np.random.choice(y, y_sample, replace=False)
    
    x_fps = [fingerprint(mol, **fingerprint_params) for mol in x]
    y_fps = [fingerprint(mol, **fingerprint_params) for mol in y]
    
    kind = 'internal' if is_internal else 'external'
    sim = np.array([[similarity(x_fp, y_fp) for x_fp in x_fps] for y_fp in y_fps])
    # delete
  #  if kind == 'internal':
  #      cond = lambda x: True
  #  else:
  #      cond = lambda x: x > 0
  #  sim = [max(filter(cond, l)) for l in sim]
  #  sim = np.array(sim).reshape(-1, 1)

    kind = 'internal' if is_internal else 'external'
    if plot_hist:
        mean, std = sim.reshape(-1).mean(), sim.reshape(-1).std()
        median = np.median(sim.reshape(-1))
       # mode = max(set(sim.reshape(-1)), key=list(sim.reshape(-1)).count)
        print(f'Mean {kind} diversity:', mean)
        print(f'Median {kind} diversity:', median)
        print(f'Standard deviation {kind} diversity:', std)
        if plot:
#            ax = plt.hist(sim.reshape(-1), **kwargs)
            old_ax = plt.gca()
            # direct all drawing to new ax if specified
            if kwargs.get('ax'):
                plt.sca(kwargs['ax'])
            sns.kdeplot(sim.reshape(-1), shade=True, color=color, alpha=alpha, **kwargs)
            if is_internal:
                plt.title('Internal distribution of fingerprint similarities')
            else:
                plt.title('External distribution of fingerprint similarities')
            plt.axvline(mean, ls='-', c=color, alpha=0.3)
            plt.axvline(mean-std, ls='--', c=color, alpha=0.3)
            plt.axvline(mean+std, ls='--', c=color, alpha=0.3)
            
            plt.xlabel('Similarity coefficient')
            plt.sca(old_ax)
            if show:
                plt.show()
    
    kind = 'internal' if is_internal else 'external'
    
    results = {f'mean_{kind}_diversity': mean,
               f'median_{kind}_diversity': median,
               f'std_{kind}_diversity': std}
    
    return results


def get_scaffolds(data, smilesCol='smiles', molCol=None, aggregate='mean', columns=None, none_value='<None>'):
    if aggregate == 'mean':
        aggregate = np.mean
    elif aggregate == 'median':
        aggregate = np.median
    
    murcko_smiles = []
    for sm in data[smilesCol]:
        try:
            scaffold = MurckoScaffoldSmiles(smiles=sm)
        except:
            print('error caught for', sm)
            scaffold = none_value
        murcko_smiles.append(scaffold)
    #murcko_smiles = [MurckoScaffoldSmiles(sm) for sm in data[smilesCol]]
    unique_scaffolds, idxs, cts = np.unique(murcko_smiles, return_inverse=True, return_counts=True)
    
    values = []
    
    values.append(('counts', cts))
    
    if columns is not None:
        for col in columns:
            tmp = []
            for i in np.unique(idxs):
                tmp.append(aggregate(data[idxs == i][col]))
            values.append((col, tmp))
    scaffold_data = pd.DataFrame(dict(values), index=unique_scaffolds)
    
    if none_value in scaffold_data.index:
        scaffold_data.drop(none_value, inplace=True)
    
    if molCol is not None:
        scaffold_data[molCol] = [Chem.MolFromSmiles(sm) for sm in scaffold_data.index]
    return scaffold_data


def compare_scaffolds(scaffolds, baseline):
    shared_scaffolds = set(scaffolds) & set(baseline)
    novel_scaffolds = set(scaffolds) - set(baseline)
    try:
        novel_fraction = len(novel_scaffolds) / len(scaffolds)
        print('Percentage of novel scaffolds: %f%% (%d / %d)' % (100*novel_fraction,
                                                                 len(novel_scaffolds),
                                                                 len(scaffolds)))
    except ZeroDivisionError:
        novel_fraction = float('nan')  
        print('Percentage of novel scaffolds: %s (%d / %d)' % (novel_fraction,
                                                               len(novel_scaffolds),
                                                               len(scaffolds)))
    return novel_scaffolds, shared_scaffolds

from rdkit.Chem import Descriptors

def compare_libraries_by_properties(data, baseline, properties, 
                                  from_smiles=True, smilesCol='smiles',
                                  molCol=None, plot=True, show=True, **kwargs):
    if molCol is not None:
        from_smiles = False
    if 'density' not in kwargs:
        kwargs['density'] = True
    if not plot:
        show = False
    if from_smiles:
        molCol = 'molecules'
        data = data.copy()
        data[molCol] = [Chem.MolFromSmiles(sm) for sm in data[smilesCol]]
        baseline = baseline.copy()
        baseline[molCol] = [Chem.MolFromSmiles(sm) for sm in baseline[smilesCol]]
    if isinstance(properties, list):
        properties = {prop: getattr(Descriptors, prop) for prop in properties}
    property_summary = {}
    for prop, func in properties.items():
        def get_prop(mol):
            try:
                return func(mol)
            except:
                return None
        
        props = []
        data[prop] = data[molCol].apply(get_prop)
        baseline[prop] = baseline[molCol].apply(get_prop)
        data_mean = data[prop].mean()
        data_median = data[prop].median()
        data_std = data[prop].std()
        baseline_mean = baseline[prop].mean()
        tot_std = pd.concat((data[prop], baseline[prop])).std()
        effect = (data_mean - baseline_mean) / tot_std
        
        print('Mean %s: %f' % (prop, data_mean))
        print('Median %s: %f' % (prop, data_median))
        print('Std %s: %f' % (prop, data_std))
        print('Effect %s: %f' % (prop, effect))
        if plot:
            fig, ax = plt.subplots()
  #          baseline[prop].plot(kind='kde', color='grey', alpha=0.2, ax=ax)#, **kwargs)
   #         data[prop].plot(kind='kde', color='red', alpha=0.2, ax=ax)#, **kwargs)
            try:
                sns.kdeplot(baseline[prop], color='grey', shade=True, alpha=0.3, label='Baseline %s' % prop)
            except:
                sns.kdeplot(baseline[prop], color='grey', shade=True, alpha=0.3, label='Baseline %s' % prop, bw=0.2)
            sns.kdeplot(data[prop], color='red', shade=True, alpha=0.3, label='Generated %s' % prop)
            plt.axvline(baseline_mean, ls='-', c='k', alpha=0.3)
            plt.axvline(data_mean, ls='-', c='r', alpha=0.3)
            plt.axvline(data_mean-data_std, ls='--', c='r', alpha=0.3)
            plt.axvline(data_mean+data_std, ls='--', c='r', alpha=0.3)
            ax.set_title(f'Distribution of {prop}')
        
        property_summary[f'mean_{prop}'] = data_mean
        property_summary[f'median_{prop}'] = data_median
        property_summary[f'std_{prop}'] = data_std
        property_summary[f'effect_{prop}'] = effect
    if show:
        plt.show()
    return property_summary

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
        self.log_results = pd.DataFrame(log_metrics, index=log_params)

    def mine_file(self, ma_window=None):
        f = open(self.filename)
        # write template for epoch data
        template = '''[\s\S]*?(?P<epoch>[0-9]+) Training on (?P<n_instances>[0-9]+) replay instances...
Setting threshold to (?P<thresholds>[0-9.]+)
Policy gradient...
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
                        for key in ('losses', 'rewards',
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

    '''
    def sample_smiles(self, epoch, kind='generated'):
        assert kind in ('max', 'cluster', 'generated')
        if kind == 'max':
            key = 'max_samples'
        elif kind == 'cluster':
            key = 'cluster_samples'
        elif kind == 'generated':
            key = 'samples'
        if kind == 'cluster':
            return list(self.loc[epoch][key]['smiles'])
        else:
            return list(self.loc[epoch][key])
    
    def smiles(self, kind='generated'):
        assert kind in ('max', 'cluster', 'generated')
        if kind == 'max':
            key = 'max_samples'
        elif kind == 'cluster':
            key = 'cluster_samples'
        elif kind == 'generated':
            key = 'samples'
        if kind == 'cluster':
            return self[key].apply(lambda row: row['smiles'])
        else:
            return self[key]

    def show_sample_mols(self, epoch, kind='generated', show=True, **kwargs):
        smiles = self.sample_smiles(epoch, kind=kind)
        if not smiles:
            return
        plt.figure()
        mols = [Chem.MolFromSmiles(sm) for sm in smiles]
        if kind == 'cluster':
            cluster_samples = self.loc[epoch]['cluster_samples']
            labels = ['%d (n=%d): %.3f' % (val, num, pred) 
                      for val, num, pred in zip(cluster_samples['clusters'],
                                                cluster_samples['sizes'],
                                                cluster_samples['predictions'])]
            im = np.array(Chem.Draw.MolsToGridImage(mols, legends=labels, **kwargs))
        else:
            im = np.array(Chem.Draw.MolsToGridImage(mols))
        plt.imshow(im)
        if show:
            plt.show()
    '''
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
    read_path = '../project/benchmark/ReLeaSE/replay_ratio201014.log'
    read_path = '../project/benchmark/ReLeaSE/n_iterations201031.log'
    lm = LogMiner(read_path, 3)
    for run in lm.runs():
        scatter_matrix(run)
    plt.show()
    #a,b,c = mine_file(read_path)
#    print(a)
 #   print(b)
  #  print(c)
    if False:
        log_miner = LogMiner(read_path, ma_window=10)
        print('LogMiner name: %s' % log_miner)
        print('displaying runs...')
        for run in log_miner.runs():
            print(run)
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
        
        mol_data = pd.read_csv('../project/datasets/cdk1_clf_augmented_ReLeaSE(2).csv')
        mol_data = pd.read_csv('../project/benchmark/ReLeaSE/replay_ratio201006-1.smi', names=['smiles','predictions'])
        exp_data = pd.read_csv('../project/datasets/CDK1_data.smi', names=['smiles', 'predictions'])
        exp_actives = exp_data.copy()[exp_data.predictions > 0.75]
        compare_libraries(mol_data, exp_data, properties=['MolWt', 'MolLogP'])
