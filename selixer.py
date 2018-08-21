# wrapper for SLURM call to elixer
from __future__ import print_function
import sys
import os
import errno
import elixer
import numpy as np
from math import ceil


#todo: allow user to change the run-time from command line
MAX_TASKS = 640 #max allowed by TACC (for gpu or vis)
time = "00:59:59"
time_set = False
email = "##SBATCH --mail-user \n\
##SBATCH --mail-type all"
queue = "vis"
tasks = 1
args = list(map(str.lower,sys.argv)) #python3 map is no longer a list, so need to cast here


#check for --merge (if so just call elixer
if "--merge" in args:
    print("Calling ELiXer to merge catalogs and fiber files (ignoring all other parameters) ... ")
    elixer.merge()
    exit(0)



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
        time_set = True
    except:
       pass
else:
    pass

#sanity check the time ... might be just hh:mm
#count the colons
colons = len(time.split(":"))-1
if colons == 1:
    time += ":00" #add seconds
elif colons != 2:
    print("Error! Invalid --time parameter")
    exit(-1)

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


#check for tasks (optional)
i = -1
if "--tasks" in args:
    i = args.index("--tasks")

if i != -1:
    try:
        tasks = int(sys.argv[i + 1])
    except:
       print("Exception parsing --tasks")

    if tasks == 0:
        print("Auto set maximum tasks ...")
        if not time_set:
            time = "00:05:00" #5 minutes (no known elixer call takes more than 1 minute
    elif (tasks < 1) or (tasks > MAX_TASKS):
        print ("Invalid --tasks value. Must be 0 (auto-max) or between 1 to 640 inclusive.")
        exit(-1)
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

### elixer.run
path = os.path.join(os.path.dirname(sys.argv[0]),"elixer.py")

if tasks == 1:
    run = "python " + path + ' ' + ' ' + ' '.join(sys.argv[1:]) + ' -f \n'

    try:
        f = open("elixer.run", 'w')
        f.write(run)
        f.close()
    except:
        print("Error! Cannot create elixer.slurm")
        exit(-1)
else: # multiple tasks
    try:

        args = elixer.parse_commandline(auto_force=True)
        print("Parsing directories to process. This may take a little while ... ")
        subdirs = elixer.get_fcsdir_subdirs_to_process(args)
        if tasks != 0:
            if tasks > len(subdirs):  # problem too many tasks requestd
                print("Error! Too many tasks (%d) requested. Only %d directories to process." % (tasks, len(subdirs)))
                exit(-1)
        else:
            tasks = min(MAX_TASKS,len(subdirs))
            print("%d tasks" % tasks)

        dirs_per_file = len(subdirs) // tasks  # int(floor(float(len(subdirs)) / float(tasks)))
        remainder = len(subdirs) % tasks
        jobs_per_task = np.full(tasks,dirs_per_file)
        jobs_per_task[0:remainder] += 1 #add one more per task to cover the remainder

        f = open("elixer.run", 'w')

        start_idx = 0
        for i in range(int(tasks)):
            fn = "dispatch_" + str(i).zfill(3)

            if not os.path.isdir(fn):
                try:
                    os.makedirs(fn)
                except OSError as exception:
                    if exception.errno != errno.EEXIST:
                        print ("Fatal. Cannot create output directory: %s" % fn)
                        exit(-1)

            df = open(os.path.join(fn,fn), 'w')
            #content = ""

            #start_idx = i * dirs_per_file
            stop_idx = start_idx + jobs_per_task[i] #min(start_idx + dirs_per_file,len(subdirs))
            for j in range(start_idx,stop_idx):
                df.write(subdirs[j] + "\n")

            df.close()

            start_idx = stop_idx #start where we left off

            #add  dispatch_xxx
            #run = "python " + path + ' ' + ' ' + ' '.join(sys.argv[1:]) + ' --dispatch ' + os.path.join(basename,fn) + ' -f \n'
            run = "cd " + fn + " ; python " + path + ' ' + ' ' + ' '.join(sys.argv[1:]) + ' --dispatch ' + fn + ' -f ; cd .. \n'
            f.write(run)

        f.close()
    except:
        print("Error! Cannot create dispatch files.")
        exit(-1)


### elixer.slurm

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
#SBATCH -n " + str(tasks) + "                  # Total number of tasks\n\
#SBATCH -p " + queue +"                 # Queue name\n\
#SBATCH -o ELIXER.o%j          # Name of stdout output file (%j expands to jobid)\n\
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
export CONTROL_FILE=elixer.run\n\
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
    f = open("elixer.slurm", 'w')
    f.write(slurm)
    f.close()
except:
    print("Error! Cannot create elixer.slurm")
    exit(-1)




# ### elixer.run
# path = os.path.join(os.path.dirname(sys.argv[0]),"elixer.py")
#
# if tasks == 1:
#     run = "python " + path + ' ' + ' ' + ' '.join(sys.argv[1:]) + ' -f \n'
#
#     try:
#         f = open("elixer.run", 'w')
#         f.write(run)
#         f.close()
#     except:
#         print("Error! Cannot create elixer.slurm")
#         exit(-1)
# else: # multiple tasks
#     try:
#         args = elixer.parse_commandline(auto_force=True)
#         print("Parsing directories to process. This may take a little while ... ")
#         subdirs = elixer.get_fcsdir_subdirs_to_process(args)
#         dirs_per_file = int(ceil(float(len(subdirs))/float(tasks)))
#
#         if tasks > len(subdirs): #problem too many tasks requestd
#             print("Error! Too many tasks (%d) requested. Only %d directories to process." %(tasks,len(subdirs)))
#             exit(-1)
#
#         f = open("elixer.run", 'w')
#
#         for i in range(int(tasks)):
#             fn = "dispatch_" + str(i).zfill(3)
#
#             if not os.path.isdir(fn):
#                 try:
#                     os.makedirs(fn)
#                 except OSError as exception:
#                     if exception.errno != errno.EEXIST:
#                         print ("Fatal. Cannot create output directory: %s" % fn)
#                         exit(-1)
#
#             df = open(os.path.join(fn,fn), 'w')
#             content = ""
#
#             start_idx = i * dirs_per_file
#             stop_idx = min(start_idx + dirs_per_file,len(subdirs))
#             for j in range(start_idx,stop_idx):
#                 df.write(subdirs[j] + "\n")
#
#             df.close()
#
#             #add  dispatch_xxx
#             #run = "python " + path + ' ' + ' ' + ' '.join(sys.argv[1:]) + ' --dispatch ' + os.path.join(basename,fn) + ' -f \n'
#             run = "cd " + fn + " ; python " + path + ' ' + ' ' + ' '.join(sys.argv[1:]) + ' --dispatch ' + fn + ' -f ; cd .. \n'
#             f.write(run)
#
#         f.close()
#     except:
#         print("Error! Cannot create dispatch files.")
#         exit(-1)
#
#


#execute system command
print ("Calling SLURM queue ...")
os.system('sbatch elixer.slurm')
#print ("Here we will call: sbatch elixer.slurm")





