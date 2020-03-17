#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=$1
python ~/Applications/ViraMiner/300/$2_branch.py /scratch/jsp4cu/plinko_viral/full_genome/full_genome_train/models/ViraMiner/300/models/$2 --input_path /scratch/jsp4cu/plinko_viral/full_genome/full_genome_train/models/ViraMiner/300/data/fg_300 --epochs 30 --filter_size 8 --layer_sizes 1000 --dropout 0.1 --learning_rate 0.001 --lr_decay None &> /scratch/jsp4cu/plinko_viral/full_genome/full_genome_train/models/ViraMiner/300/output/vm.300.$2.out
