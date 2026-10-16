#!/bin/bash
#SBATCH --job-name=LLMtest
#SBATCH --output=%x_%j.out    
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --time=60:00
#SBATCH --partition=gpu_a100






 
source $HOME/projectVenv/bin/activate


python $HOME/ULLMPROJ/ULLM_Project/data_preprocessing.py
python $HOME/ULLMPROJ/ULLM_Project/llm_test.py
 
deactivate