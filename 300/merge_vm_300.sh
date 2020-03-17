#!/usr/bin/env bash
# $1 cuda device
export CUDA_VISIBLE_DEVICES=$1
python ~/Applications/ViraMiner/300/merge_and_retrain.py /scratch/jsp4cu/plinko_viral/full_genome/full_genome_train/models/ViraMiner/300/models/ --input_path /scratch/jsp4cu/plinko_viral/full_genome/full_genome_train/models/ViraMiner/300/data/fg_300  --pattern_model /scratch/jsp4cu/plinko_viral/full_genome/full_genome_train/models/ViraMiner/300/models/pattern.hdf5 --freq_model /scratch/jsp4cu/plinko_viral/full_genome/full_genome_train/models/ViraMiner/300/models/frequency.hdf5 --finetuning True --epochs 30 --dropout 0.1 --learning_rate 0.001 --lr_decay None &> /scratch/jsp4cu/plinko_viral/full_genome/full_genome_train/models/ViraMiner/300/output/vm.300.merge.out
