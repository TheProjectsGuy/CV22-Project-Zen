#!/bin/bash
#SBATCH -A research
#SBATCH -J "pix2pix-v2"
#SBATCH -c 10
#SBATCH -G 1
#SBATCH --mem-per-cpu=4G
#SBATCH --exclude="gnode[03-42,90-92]"
#SBATCH -o "pix2pix-r2.txt"
#SBATCH --time="1-12:00:00"
#SBATCH --mail-type=END


# ======================================================
# SLURM script for training Pix2Pix on Ada
# ======================================================

echo "[BLOCK] ======= Inspecting node ======="
echo "Host: $HOSTNAME"
source $HOME/.bashrc    # Environment script
echo ""


echo "[BLOCK] ======= Loading data into the node ======="
# Check if scratch folder exists
scratch_dir="/scratch/$USER"
if [ ! -d $scratch_dir ]; then
    mkdir $scratch_dir
fi

# Move data from /share to /scratch
data_local_dir="/share1/$USER/datasets/cityscapes.tar.gz"
scp $USER@ada:$data_local_dir $scratch_dir
echo "Dataset moved to $HOSTNAME"
echo ""


echo "[BLOCK] ======= Setting up node environment ======="
# Unzip everything
cd $scratch_dir
tar -xf ./cityscapes.tar.gz
echo "Unzip successful (pwd: `pwd`)"
# Load modules
module load cuda/10.2
module load cudnn/7.6.5-cuda-10.2
module list
echo "Modules successfully loaded"
# Anaconda environment
conda-init
conda activate pix2pix-torch

echo "[BLOCK] ======= Main training code ======="
# Variables for script
scratch_dir="/scratch/$USER/cityscapes/"    # Again (sanity check)
out_dir="$scratch_dir/pix2pix/"
if [ ! -d $out_dir ]; then
    mkdir $out_dir
fi

data_dir="$scratch_dir"
num_epochs=200
ckpt_freq=50
# Main function call
echo "[BLOCK] ======== Starting Training =========="
python ~/Documents/pix2pix/pix2pix_custom.py --data-dir=$data_dir \
    --out-dir=$out_dir --data-seg=$data_seg --num-epochs=$num_epochs \
    --epoch-ckpt-freq=$ckpt_freq
# Save everything - out_dir exists (script creates it)
if [ ! -d $out_dir ]; then
    echo "[ERROR] Output directory '$out_dir' does not exist, no backup"
else
    # Make backup
    cd $out_dir
    echo "In `pwd` (saving everything here)"
    tar -cvf ./slurm_res_$SLURM_JOB_ID.tar ./ # Backup current dir
    echo "Archive created"
    ls -laR `pwd`
    # Transfer backup
    scp ./slurm_res_$SLURM_JOB_ID.tar $USER@ada:/share1/$USER
    echo "Backup transferred to '/share1/$USER'"
fi
echo ""


echo "[BLOCK] ======= End of script ======="
echo "Deleting $scratch_dir"
rm -rf $scratch_dir
echo "Script has ended"