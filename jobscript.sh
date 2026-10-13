#!/bin/bash
#SBATCH --job-name=LLMtest
#SBATCH --output=%x_%j.out    
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=30:00
#SBATCH --partition=gpu_mig
#SBATCH --mem-per-gpu=36G






 
source $HOME/projectVenv/bin/activate



python $HOME/ULLMPROJ/ULLM_Project/llm_test.py
 
deactivate