#!/bin/bash 
#SBATCH --job-name=exp_misr_joint_srdiff_lcc_test_base_ddpm_16_clip
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=5G
#SBATCH --time=48:00:00
#SBATCH --mail-user=kanyamahanga@ipi.uni-hannover.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output logs/exp_misr_joint_srdiff_lcc_test_base_ddpm_16_clip_%j.out
#SBATCH --error logs/exp_misr_joint_srdiff_lcc_test_base_ddpm_16_clip_%j.err
source load_modules.sh
export CONDA_ENVS_PATH=$HOME/.conda/envs
export DATA_DIR=$BIGWORK
conda activate /software/NHGN20600/nhgnkany/flair_venv
which python
cd $HOME/MISR_JOINT_SRDiff_LCC_BASE_DDPM_16_CLIP
srun python trainer.py --config configs/diffsr_maxvit_ltae.yaml --config_file flair-config-server.yml --exp_name srdiff_maxvit_ltae_ckpt --hparams="diff_net_ckpt=/bigwork/nhgnkany/Results/MISR_JOINT_SRDiff_LCC_BASE_DDPM_16_CLIP/checkpoints/srdiff_maxvit_ltae_ckpt" --infer
