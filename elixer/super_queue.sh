# much the same as super_slurm.sh
# this one, though, jumps into each directory and queues the slurm jobs
# applying a dependency on each prior job

#polonius@ls6:/c00/$ sbatch --dependency afterany:1045032 elixer.slurm
#
#-----------------------------------------------------------------
#           Welcome to the Lonestar6 Supercomputer
#-----------------------------------------------------------------
#
#No reservation for this job
#--> Verifying valid submit host (login1)...OK
#--> Verifying valid jobname...OK
#--> Verifying valid ssh keys...OK
#--> Verifying access to desired queue (normal)...OK
#--> Checking available allocation (AST23008)...OK
#--> Verifying that quota for filesystem /home1/03261/polonius is at 63.09% allocated...OK
#--> Verifying that quota for filesystem /work/03261/polonius/ls6 is at 83.40% allocated...OK
#Submitted batch job 1046944



#todo: need to get batch job ID from previous sbatch call so can feed in as a dependency for the next call

#dets=$(ls [cd]??_*[0-9].dets)
#lastjob=0
#slurm="elixer.slrum"
#
#for det in $dets
#do
#  cd ${det}
#  if (())${lastjob} == 0)); then
#    lastjob=$(sbatch ${elixer.slurm})
#  else
#    sbatch --dependency afterany:${lastjob} ${elixer.slurm}
#  fi
#  cd ..
#done