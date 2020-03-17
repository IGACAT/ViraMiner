#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=$1
python ~/Applications/ViraMiner/$2_branch.py /scratch/jsp4cu/plinko_viral/full_genome/full_genome_train/models/50/viraminer/models/$2 --input_path /scratch/jsp4cu/plinko_viral/full_genome/full_genome_train/models/50/viraminer/data/fg_50 --epochs 30 --filter_size 8 --layer_sizes 1000 --dropout 0.1 --learning_rate 0.001 --lr_decay None &> /scratch/jsp4cu/plinko_viral/full_genome/full_genome_train/models/50/viraminer/output/vm.100.50.$2.out
