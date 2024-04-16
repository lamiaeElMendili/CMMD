#!/bin/bash
#SBATCH --gres=gpu:1 
#SBATCH --cpus-per-task=12
#SBATCH --account=def-sdaniel
#SBATCH --mem=120000M      
#SBATCH --time=01-18:00   # DD-HH:MM:SS
#SBATCH --mail-user=lamiae.el-mendili.1@ulaval.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END                                                                                                         
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

module load StdEnv/2020  gcc/9.3.0
module load cuda/11.1
module load python/3.8.2
module load openblas
virtualenv --no-download $SLURM_TMPDIR/pcl
source $SLURM_TMPDIR/pcl/bin/activate

pip install --upgrade pip setuptools wheel
pip install --no-index --upgrade pip
pip install --no-index torch==1.12.0
pip install --no-index ninja
pip install --no-index -U MinkowskiEngine  -v --no-deps
pip install --no-index scipy

pip install --no-index plyfile
pip install --no-index scikit-learn==0.24.2 
pip install --no-index tqdm                                                                                                                                             
pip install --no-index matplotlib 
pip install --no-index open3d==0.12.0 
pip install --no-index seaborn
pip install --no-index numpy 
pip install --no-index omegaconf
pip install --no-index --upgrade vispy







pip install pytorch-lightning==1.5.0 --no-index

pip install --no-index wandb

wandb login 1452c1f31edcc38f0d34fc5ab926714acf4c4df9

pip uninstall opencv-python -y
pip install opencv-python>=4.5.4.58 --no-index
pip install --no-index nuscenes_devkit-1.1.9-py3-none-any.whl


cd CMMD

#python adapt_cosmix.py --config_file configs/adaptation/synlidar2semanticposs_cosmix.yaml --method cmmd-cosmix

python adapt_cosmix.py --config_file configs/adaptation/nuscenes2semanticposs/nuscenes2semanticposs_gaussian.yaml --method cmmd-cosmix
#python train_source.py --config_file configs/source/nuscenes2semanticposs.yaml
# to sync cosmix run 
# wandb sync /home/lamiaeel/projects/def-sdaniel/lamiaeel/CMMD/wandb/offline-run-20240405_113129-55l2yo7s -p 'SynLiDAR->SemanticKITTI'


