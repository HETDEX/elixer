# wrapper for SLURM call to voltron
from __future__ import print_function
import sys
import os
import errno


#todo: allow user to change the run-time from command line
time = "00:59:59"
email = "##SBATCH --mail-user \n\
##SBATCH --mail-type all"
queue = "vis"
args = list(map(str.lower,sys.argv)) #python3 map is no longer a list, so need to cast here

#check for name agument (mandatory)
i = -1
if "-n" in args:
    i = args.index("-n")
elif "--name" in args:
    i = args.index("--name")

if i != -1:
    try:
        basename = sys.argv[i+1]
    except:
        print ("Error! Cannot find mandatory parameter --name")
        exit(-1)
else:
    print("Error! Cannot find mandatory parameter --name")
    exit(-1)


#check for time argument (optional)
i = -1
if "-t" in args:
    i = args.index("-t")
elif "--time" in args:
    i = args.index("--time")

if i != -1:
    try:
        time = sys.argv[i + 1]
    except:
       pass
else:
    pass

#check for email argument (optional)
i = -1
if "--email" in args:
    i = args.index("--email")

if i != -1:
    try:
        email_addr = sys.argv[i + 1]
        if (email_addr is not None) and ('@' in email_addr) and (len(email_addr) > 5):
            #assume good
            email = "#SBATCH --mail-user " + email_addr + "\n#SBATCH --mail-type all"
    except:
       pass
else:
    pass



#check for queue (optional)
i = -1
if "--queue" in args:
    i = args.index("--queue")

if i != -1:
    try:
        queue = sys.argv[i + 1]
    except:
       pass
else:
    pass

if not os.path.isdir(basename):
    try:
        os.makedirs(basename)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            print("Error! Cannot create output directory: %s" % basename)
            exit(-1)

os.chdir(basename)


### voltron.slurm

slurm = "\
#!/bin/bash \n\
#\
# Simple SLURM script for submitting multiple serial\n\
# jobs (e.g. parametric studies) using a script wrapper\n\
# to launch the jobs.\n\
#\n\
# To use, build the launcher executable and your\n\
# serial application(s) and place them in your WORKDIR\n\
# directory.  Then, edit the CONTROL_FILE to specify \n\
# each executable per process.\n\
#-------------------------------------------------------\n\
#-------------------------------------------------------\n\
# \n\
#------------------Scheduler Options--------------------\n\
#SBATCH -J HETDEX              # Job name\n\
#SBATCH -n 1                  # Total number of tasks\n\
#SBATCH -p " + queue +"                 # Queue name\n\
#SBATCH -o VOLTRON.o%j          # Name of stdout output file (%j expands to jobid)\n\
#SBATCH -t " + time + "            # Run time (hh:mm:ss)\n\
#SBATCH -A Hobby-Eberly-Telesco\n"\
+ email + "\n\
#------------------------------------------------------\n\
#\n\
# Usage:\n\
#	#$ -pe <parallel environment> <number of slots> \n\
#	#$ -l h_rt=hours:minutes:seconds to specify run time limit\n\
# 	#$ -N <job name>\n\
# 	#$ -q <queue name>\n\
# 	#$ -o <job output file>\n\
#	   NOTE: The env variable $JOB_ID contains the job id. \n\
#\n\
#------------------------------------------------------\n\
\n\
#------------------General Options---------------------\n\
module unload xalt \n\
module load launcher\n\
export EXECUTABLE=$TACC_LAUNCHER_DIR/init_launcher\n\
export WORKDIR=. \n\
export CONTROL_FILE=voltron.run\n\
\n\
# Variable descriptions:\n\
#\n\
#  TACC_LAUNCHER_PPN = number of simultaneous processes per host\n\
#                      - if this variable is not set, value is\n\
#                        determined by the process density/wayness\n\
#                        specified in 'Scheduler Options'\n\
#  EXECUTABLE        = full path to the job launcher executable\n\
#  WORKDIR           = location of working directory\n\
#  CONTROL_FILE      = text input file which specifies\n\
#                      executable for each process\n\
#                      (should be located in WORKDIR)\n\
#------------------------------------------------------\n\
\n\
#--------- Intel Xeon Phi Options (EXPERIMENTAL) -------------\n\
export TACC_LAUNCHER_NPHI=0\n\
export TACC_LAUNCHER_PHI_PPN=8\n\
export PHI_WORKDIR=.\n\
export PHI_CONTROL_FILE=phiparamlist\n\
\n\
# Variable descriptions:\n\
#  TACC_LAUNCHER_NPHI    = number of Intel Xeon Phi cards to use per node\n\
#                          (use 0 to disable use of Xeon Phi cards)\n\
#  TACC_LAUNCHER_PHI_PPN = number of simultaneous processes per Xeon Phi card\n\
#  PHI_WORKDIR           = location of working directory for Intel Xeon Phi jobs\n\
#  PHI_CONTROL_FILE      = text input file which specifies executable\n\
#                          for each process to be run on Intel Xeon Phi\n\
#                          (should be located in PHI_WORKDIR)\n\
#------------------------------------------------------\n\
\n\
#------------ Task Scheduling Options -----------------\n\
export TACC_LAUNCHER_SCHED=interleaved\n\
\n\
# Variable descriptions:\n\
#  TACC_LAUNCHER_SCHED = scheduling method for lines in CONTROL_FILE\n\
#                        options (k=process, n=num. lines, p=num. procs):\n\
#                          - interleaved (default): \n\
#                              process k executes every k+nth line\n\
#                          - block:\n\
#                              process k executes lines [ k(n/p)+1 , (k+1)(n/p) ]\n\
#                          - dynamic:\n\
#                              process k executes first available unclaimed line\n\
#--------------------------------------------------------\n\
\n\
#----------------\n\
# Error Checking\n\
#----------------\n\
\n\
if [ ! -d $WORKDIR ]; then \n\
        echo \" \" \n\
	echo \"Error: unable to change to working directory.\" \n\
	echo \"       $WORKDIR\" \n\
	echo \" \" \n\
	echo \"Job not submitted.\"\n\
	exit\n\
fi\n\
\n\
if [ ! -x $EXECUTABLE ]; then\n\
	echo \" \"\n\
	echo \"Error: unable to find launcher executable $EXECUTABLE.\"\n\
	echo \" \"\n\
	echo \"Job not submitted.\"\n\
	exit\n\
fi\n\
\n\
if [ ! -e $WORKDIR/$CONTROL_FILE ]; then\n\
	echo \" \"\n\
	echo \"Error: unable to find input control file $CONTROL_FILE.\"\n\
	echo \" \"\n\
	echo \"Job not submitted.\"\n\
	exit\n\
fi\n\
\n\
#----------------\n\
# Job Submission\n\
#----------------\n\
\n\
cd $WORKDIR/ \n\
echo \" WORKING DIR:   $WORKDIR/\"\n\
\n\
$TACC_LAUNCHER_DIR/paramrun SLURM $EXECUTABLE $WORKDIR $CONTROL_FILE $PHI_WORKDIR $PHI_CONTROL_FILE\n\
\n\
echo \" \"\n\
echo \" Parameteric Job Complete\"\n\
echo \" \" "

try:
    f = open("voltron.slurm", 'w')
    f.write(slurm)
    f.close()
except:
    print("Error! Cannot create voltron.slurm")
    exit(-1)




### voltron.run
path = os.path.join(os.path.dirname(sys.argv[0]),"voltron.py")
run = "python " + path + ' ' + ' ' + ' '.join(sys.argv[1:]) + ' -f \n'

try:
    f = open("voltron.run", 'w')
    f.write(run)
    f.close()
except:
    print("Error! Cannot create voltron.slurm")
    exit(-1)


#execute system command
print ("Calling SLURM queue ...")
os.system('sbatch voltron.slurm')
#print ("Here we will call: sbatch voltron.slurm")





