import os
import sys
import argparse
import ast
from functools import partial

import time
from iter_utils import DictIterator
#print = partial(print, flush=False)

def parse_args():
    parser = argparse.ArgumentParser(description='Perform a grid search on a general .py script that takes any'
                                                 ' number of keyword arguments.')
    parser.add_argument('script', type=str,
                        help='Filename of python script to be run for gridsearch. Script must have '
                             'a main method that takes any number of keyword arguments. '
		    	     'The script may have an option to write a generated library to a file. '
                             'In this case the main method must have a "save_path" keyword argument')
    parser.add_argument('config_path', type=str,
                        help='Filename containing parameters to be passed to the running script. The parameters '
                             'should be stored as a dictionary with values of type list for parameters '
                             'to be optimized. ')
    parser.add_argument('log_path', type=str,
                        help='Filename for logging training information. All printed output of the script '
                             'will be redirected to the log_path.')
    return parser.parse_args()

def main():    
    args = parse_args()
    script = args.script
    if script.endswith('.py'):
        script = script[:-3]
    config_path = args.config_path
    if config_path.endswith('.py'):
        run_path = args.log_path[:-3]
    else:
        run_path = args.log_path
    # must add to sys path if filename
    dir_path = os.path.dirname(script)
    module = os.path.split(script)[-1]
    tmp = __import__(module, globals(), locals(), ['main'])
    main_ = tmp.main

    gridsearch_params = ast.literal_eval(open(config_path).read())

    log_path = run_path + '.log'
    with open(log_path, 'wt') as f:
        print('starting log', file=f)

    for i, params in enumerate(DictIterator(gridsearch_params)):
        save_path = run_path + '-%d.smi' % (i+1)
        params['save_path'] = save_path
        sys.stdout = open(log_path, 'a')
        try:
            print('\n')
            main_(**params)
        finally:
            sys.stdout.close()
            sys.stdout = sys.__stdout__

if __name__ == '__main__':
    main()
