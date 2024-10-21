#!/bin/sh
#SBATCH --export=ALL # export all environment variables to the batch job.
#SBATCH -p highmem # submit to the idsai GPU queue
#SBATCH --time=21-00:00:00 # Maximum wall time for the job.
#SBATCH -A research_project-idsai # research project to submit under. 
#SBATCH --nodes=1 # specify number of nodes.
#SBATCH --ntasks-per-node=32 # specify number of processors per node
#SBATCH --mem=64G # specify bytes of memory to reserve
#SBATCH --mail-type=END # send email at job completion 
#SBATCH --output=output_logs/aug_lc.o
#SBATCH --error=error_logs/aug_lc.e
#SBATCH --job-name=ak_sd_paper
#SBATCH --array=0

## print start date and time
echo Job started on:
date -u

## print node job run on
echo -n "This script is running on "
hostname

source activate torch_env
# pip install -r igpt_isca_requirements.txt

python -u calc_aug_acc_lc.py

## print which array number this is
echo "This job number " ${SLURM_ARRAY_JOB_ID}  " and array number " ${SLURM_ARRAY_TASK_ID}

## print end date and time
echo Job ended on:
date -u
