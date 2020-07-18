#!/bin/bash 
#SBATCH -J ELiXerDB              # Job name
#SBATCH -N 1                  # Total number of nodes requested
#SBATCH --ntasks-per-node 3       #Tasks per node
#SBATCH -p normal                 # Queue name
#SBATCH -o ELIXER.o%j          # Name of stdout output file (%j expands to jobid)
#SBATCH -t 20:00:00            # Run time (hh:mm:ss)
#SBATCH -A Hobby-Eberly-Telesco
#SBATCH --mail-user dustin@astro.as.utexas.edu
#SBATCH --mail-type all
module load launcher
export TACC_LAUNCHER_PPN=24
export EXECUTABLE=$TACC_LAUNCHER_DIR/init_launcher
export LAUNCHER_WORKDIR=$(pwd)
export LAUNCHER_JOB_FILE=elixer_reports.run
WORKDIR=$LAUNCHER_WORKDIR
CONTROL_FILE=$LAUNCHER_JOB_FILE
export LAUNCHER_SCHED=interleaved
export LD_PRELOAD=/work/00410/huang/share/patch/myopen.so 
#----------------
# Error Checking
#----------------

if [ ! -d $WORKDIR ]; then 
        echo " " 
	echo "Error: unable to change to working directory." 
	echo "       $WORKDIR" 
	echo " " 
	echo "Job not submitted."
	exit
fi

if [ ! -x $EXECUTABLE ]; then
	echo " "
	echo "Error: unable to find launcher executable $EXECUTABLE."
	echo " "
	echo "Job not submitted."
	exit
fi

if [ ! -e $WORKDIR/$CONTROL_FILE ]; then
	echo " "
	echo "Error: unable to find input control file $CONTROL_FILE."
	echo " "
	echo "Job not submitted."
	exit
fi

#----------------
# Job Submission
#----------------

cd $WORKDIR/ 
echo " WORKING DIR:   $WORKDIR/"

$TACC_LAUNCHER_DIR/paramrun

echo " "
echo " Parameteric Job Complete"
echo " " 
