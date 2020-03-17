#!/usr/bin/env bash
# $1 cuda device
export CUDA_VISIBLE_DEVICES=$1
python ~/Applications/ViraMiner/merge_and_retrain.py /scratch/jsp4cu/plinko_viral/full_genome/full_genome_train/models/ViraMiner/100/models/ --input_path /scratch/jsp4cu/plinko_viral/full_genome/full_genome_train/models/ViraMiner/100/data/fg_100  --pattern_model /scratch/jsp4cu/plinko_viral/full_genome/full_genome_train/models/ViraMiner/100/models/pattern.hdf5 --freq_model /scratch/jsp4cu/plinko_viral/full_genome/full_genome_train/models/ViraMiner/100/models/frequency.hdf5 --finetuning True --epochs 30 --dropout 0.1 --learning_rate 0.001 --lr_decay None &> /scratch/jsp4cu/plinko_viral/full_genome/full_genome_train/models/ViraMiner/100/output/vm.100.merge.out
