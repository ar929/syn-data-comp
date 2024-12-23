#!/bin/sh
#SBATCH --export=ALL                # Export all environment variables to the batch job
#SBATCH -p idsaivolta               # Submit to the idsai GPU queue
#SBATCH --time=21-00:00:00          # Maximum wall time for the job
#SBATCH -A research_project-idsai   # Research project to submit under
#SBATCH --nodes=1                   # Specify number of nodes
#SBATCH --ntasks-per-node=32        # Specify number of processors per node
#SBATCH --mem=64G                   # Specify memory to reserve
#SBATCH --mail-type=END             # Send email at job completion
#SBATCH --output=output_logs/igpt_data_sorting.o
#SBATCH --error=error_logs/igpt_data_sorting.e
#SBATCH --job-name=igpt_data_sorting
#SBATCH --array=0                   # Job array specification

#SBATCH --dependency=afterok:1208478_{1..15}  # Wait for all job array tasks

# Paths
SOURCE_FOLDER="${HOME}/syn_data/synthetic_output"
SEED_42_FOLDER="${HOME}/syn_data/igpt_data/synthetic_cifar_seed_42"
OTHER_FILES_FOLDER="${HOME}/syn_data/igpt_data/synthetic_cifar_43_to_47"

# Create target directories if they don't exist
mkdir -p "$SEED_42_FOLDER"
mkdir -p "$OTHER_FILES_FOLDER"

# Sort and copy files (only process .png files)
echo "Sorting .png files from source folder: $SOURCE_FOLDER"
find "$SOURCE_FOLDER" -type f -name '*seed_42*.png' -exec cp {} "$SEED_42_FOLDER" \;
find "$SOURCE_FOLDER" -type f -name '*.png' ! -name '*seed_42*.png' -exec cp {} "$OTHER_FILES_FOLDER" \;

# Zip the folder containing files other than 'seed_42'
echo "Zipping other files folder: $OTHER_FILES_FOLDER"
zip -r "${OTHER_FILES_FOLDER}.zip" "$OTHER_FILES_FOLDER"

echo "File sorting and zipping completed successfully!"
