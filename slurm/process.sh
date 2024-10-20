#!/bin/bash
#SBATCH --job-name=process_train      # Job name
#SBATCH --output=/scratch/dkayande/process/slurm/process.%j.out    # Standard output log
#SBATCH --error=/scratch/dkayande/process/slurm/process.%j.err     # Standard error log

#SBATCH --partition=gpu-a100                    # Partition (queue)
#SBATCH --time=01:00:00                        # Runtime limit (hh:mm:ss)
#SBATCH --ntasks=1                             # Number of tasks (1 because it's a single job)
#SBATCH --cpus-per-task=1                     # Number of CPUs per task
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=16G                       # Memory per CPU (32GB total for 8 CPUs)



module use /apps/generic/modulefiles 
module load miniconda3                   
conda activate /home/dkayande/.conda/envs/process                 

cd /scratch/dkayande/process
previous=$(/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/tail -n '+2')

srun python main.py

/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/grep -v -F "$previous"
