#!/bin/bash 
# #SBATCH --job-name=exp_srdiff_test_inference_joint_srdiff_lcc_maxvit_hr5_convformer_sr4_nir_caf_focal_all_eolab
# #SBATCH --partition=gpu
# #SBATCH --nodes=1
# #SBATCH --ntasks-per-node=1
# #SBATCH --gres=gpu:a100m40:1
# #SBATCH --cpus-per-task=8
# #SBATCH --mem-per-cpu=4G
# #SBATCH --time=48:00:00
# #SBATCH --mail-user=kanyamahanga@ipi.uni-hannover.de
# #SBATCH --mail-type=BEGIN,END,FAIL
# #SBATCH --output logs/exp_srdiff_test_inference_joint_srdiff_lcc_maxvit_hr5_convformer_sr4_nir_caf_focal_all_eolab_%j.out
# #SBATCH --error logs/exp_srdiff_test_inference_joint_srdiff_lcc_maxvit_hr5_convformer_sr4_nir_caf_focal_all_eolab_%j.err
# source load_modules.sh

export CONDA_ENVS_PATH=$HOME/.conda/envs
DATA_DIR="/my_data/"
export DATA_DIR
source /home/eouser/flair_venv/bin/activate
which python
cd $HOME/exp_2026/MISR_JOINT_SRDiff_LCC_BASE_DDPM_16_CLIP
python trainer.py --config configs/diffsr_maxvit_ltae.yaml --config_file flair-config-server-eolab.yml --exp_name misr/srdiff_maxvit_ltae_ckpt --hparams="diff_net_ckpt=/my_data/Results/MISR_JOINT_SRDiff_LCC_BASE_DDPM_16_CLIP/checkpoints/srdiff_maxvit_ltae_ckpt" --infer
