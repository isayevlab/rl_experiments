#!/bin/bash
seeds=(1 2 3)
for s in ${seeds[@]}
do
	bash set_seed.sh $s
	bash train.sh
done
