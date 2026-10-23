#!/bin/bash
#SBATCH --job-name=LLMtest
#SBATCH --output=%x_%j.out    
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --time=02:00:00
#SBATCH --partition=gpu_h100






 
source $HOME/projectVenv/bin/activate


#python $HOME/ULLMPROJ/ULLM_Project/data_preprocessing.py
#python $HOME/ULLMPROJ/ULLM_Project/data_preprocess_fever_orig.py
python $HOME/ULLMPROJ/ULLM_Project/llm_test.py llama
python $HOME/ULLMPROJ/ULLM_Project/llm_test.py oss
python $HOME/ULLMPROJ/ULLM_Project/llm_test.py qwen 
deactivate