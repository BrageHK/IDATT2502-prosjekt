#!/bin/bash
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:p100:1
#SBATCH --account=share-ie-idi
#SBATCH --time=0:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=12000
#SBATCH --job-name="Training AplhaZero"
#SBATCH --output=test-srun.out
#SBATCH --mail-user=bragehk@stud.ntnu.no
#SBATCH --mail-type=ALL


WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "we are running from this directory: $SLURM_SUBMIT_DIR"
echo " the name of the job is: $SLURM_JOB_NAME"
echo "Th job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"

module purge
module load Python/3.9.5-GCCcore-10.3.0

cd IDATT2502-prosjekt

pip3 install --user -r requirements.txt

cd src

echo "Running Training.py"
python3 Training.py
echo "Done!"