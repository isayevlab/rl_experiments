files=('timelapse' 
       'mixed_test' 'n_fine_tune'
       'primed_model' 'replay_data'
       'replay_combo' 'replay_ratio'
       'replay_ratio_mixed')
seed=$1
echo 'setting seed to '$1
for f in ${files[@]}
do
#	config_src=$(echo ../project/config/backup/$f.txt)
	config=$(echo ../config/$f.txt)
	sed -i -e "s/'seed': [0-9]\+/'seed': "$seed"/g" $config
done
