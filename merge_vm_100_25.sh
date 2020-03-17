#!/usr/bin/env bash
# $1 cuda device
export CUDA_VISIBLE_DEVICES=$1
python ~/Applications/ViraMiner/merge_and_retrain.py /scratch/jsp4cu/plinko_viral/full_genome/full_genome_train/models/50/viraminer/models/merge  --input_path /scratch/jsp4cu/plinko_viral/full_genome/full_genome_train/models/50/viraminer/data/fg_50 --pattern_model /scratch/jsp4cu/plinko_viral/full_genome/full_genome_train/models/50/viraminer/models/pattern.hdf5  --freq_model /scratch/jsp4cu/plinko_viral/full_genome/full_genome_train/models/50/viraminer/models/frequency.hdf5  --finetuning True --epochs 30 --dropout 0.1 --learning_rate 0.001 --lr_decay None &> /scratch/jsp4cu/plinko_viral/full_genome/full_genome_train/models/50/viraminer/output/vm.100.50.merge.out
