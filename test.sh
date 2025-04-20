#!/bin/bash
#SBATCH -J lab3_dedicated
#SBATCH --partition ice-cpu,coc-cpu
#SBATCH -C "intel&core24"
#SBATCH -N 4 --ntasks-per-node=16
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out

#Setup Environment
cd $SLURM_SUBMIT_DIR
source pace_env.sh

make teragen terametrics

#List of test sizes (all ≥ 64 procs)
sizes=(64 67 69 125 2000 344230 78 1023 10000 16000000)

for i in "${!sizes[@]}"
do
    SIZE=${sizes[$i]}
    echo "* Test $((i+1)): Generating $SIZE records"

    rm -f data_bench.dat

    mpirun -np 1 ./teragen -c $SIZE -f data_bench.dat
    sleep 2

    echo "* Running terametrics on $SIZE records"
    mpirun ./terametrics -f data_bench.dat -c 20
    echo "===================================================="
done

exit 0