#!/bin/bash

f='timelapse'
config=$(echo ../config/$f.txt)
echo 'using config from: '$config
savepath=$(echo '../logs/'$f$(date +'%y%m%d'))
echo 'saving to: '$savepath
echo
python -m gridsearch egfr_demo_timelapse $config $savepath

files=('mixed_test' 'n_fine_tune'
       'primed_model' 'replay_data'
       'replay_combo' 'replay_ratio'
       'replay_ratio_mixed')
#files=('replay_ratio' 'replay_ratio_mixed')
for f in ${files[@]}
do
	config=$(echo '../config/'$f'.txt')
	echo 'using config from: '$config
	savepath=$(echo '../logs/'$f$(date +'%y%m%d'))
	echo 'saving to: '$savepath
	echo
	python -m gridsearch egfr_demo $config $savepath
done
