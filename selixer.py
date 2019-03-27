# wrapper for SLURM call to elixer
from __future__ import print_function
import sys
import os
import errno
try:
    from elixer import elixer
except:
    import elixer

import numpy as np
from math import ceil
from datetime import timedelta
import socket

#python version
PYTHON_MAJOR_VERSION = sys.version_info[0]
PYTHON_VERSION = sys.version_info

#there is an issue with multiple pythons on some TACC systems, we want to run with whatever version
#called selixer ... which might not otherwise be the same when SLURM runs
try:
    python_cmd = sys.executable + " "
except:
    python_cmd = "python "

pre_python_cmd = ""

hostname = socket.gethostname()
#print("+++++++++++++ put this BACK !!!!! ")
#hostname = "wrangler"

HOST_LOCAL = -1
HOST_UNKNOWN = 0
HOST_MAVERICK = 1
HOST_WRANGLER = 2
HOST_STAMPEDE2 = 3

host = HOST_UNKNOWN


# def remove_ra_dec(arg_list):
#     new_list = []
#     skip_next = False
#     for arg in arg_list:
#         arg = str(arg).lower()
#         if  (arg == "--ra") or (arg == "--dec"):
#             skip_next = True
#
#         elif not skip_next:
#             new_list.append(arg)
#         else:
#             skip_next = False
#     return new_list

### NAMING, NOTATION
# job == an elixer dispatch
# task == on the cluster ...
#         one line in the execution file [elixer.run]
#         becomes the number of dispatch_xxx subdirectories created
#         each will have one or more explicitly listed detections to process (each line in the dispatch_xxx file)
# core == single processing unit (runs one elixer call)
# node == group of cores (shares memory and other resources)


#form of loginxxx.<name>.tacc.utexas.edu for login nodes
#or cxxx-xxx.<name>.tacc.utexas.edu for compute nodes

args = list(map(str.lower,sys.argv)) #python3 map is no longer a list, so need to cast here

#check for --merge (if so just call elixer
if "--merge" in args:
    print("Calling ELiXer to merge catalogs and fiber files (ignoring all other parameters) ... ")
    elixer.merge()
    exit(0)


if "--recover" in args:
    recover_mode = True
else:
    recover_mode = False

#check for queue (optional)
queue = None
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


if "tacc.utexas.edu" in hostname:
    hostname = hostname.split(".")[1]

if hostname == "maverick":
    print("preparing SLURM for maverick...")
    host = HOST_MAVERICK
    MAX_TASKS = 640 #max allowed by TACC (for gpu or vis)
    TIME_OVERHEAD = 2.0 #MINUTES of overhead to get started (per task call ... just a safety)
    if recover_mode:
        MAX_TIME_PER_TASK = 1.5 #in recover mode, can bit more agressive in timing (easier to continue if timeout)
    else:
        MAX_TIME_PER_TASK = 3.0  # MINUTES max, worst case expected time per task to execute (assumes minimal retries)
    cores_per_node = 20 #for Maverick
    time = "00:59:59"
    time_set = False
    email = "##SBATCH --mail-user\n##SBATCH --mail-type all"
    queue = "vis"
    tasks = 1
elif hostname == "wrangler":
    #!!! right now, for whatever reason, memmory is a problem for wrangler
    #and it can handle only 4 tasks per node (maximum)
    #It takes about 1 minute to run a task (lets call it 90 seconds to be safe)
    #set -N and -n s|t n/N <= 4
    print("preparing SLURM for wrangler...")
    host = HOST_WRANGLER
    MAX_TASKS = 10
    MAX_NODES = 1
    #MAX_TASKS_PER_NODE = 10 #actually, variable, encoded later
    TIME_OVERHEAD = 2.0  # MINUTES of overhead to get started (per task call ... just a safety)
    if recover_mode:
        MAX_TIME_PER_TASK = 1.5 #in recover mode, can bit more agressive in timing (easier to continue if timeout)
    else:
        MAX_TIME_PER_TASK = 3.0  # MINUTES max, worst case expected time per task to execute (assumes minimal retries)

    cores_per_node = 24

    if PYTHON_MAJOR_VERSION < 3:
        MAX_TASKS = 300  # point of seriously diminishing returns
        MAX_NODES = 50  # right now, pointless to go beyond 2 nodes
        MAX_TASKS_PER_NODE = 6  # actually, variable, encoded later
    else:
        MAX_TASKS = 2400
        MAX_NODES = 100
        MAX_TASKS_PER_NODE = 24

    time = "00:59:59"
    time_set = False
    email = "##SBATCH --mail-user\n##SBATCH --mail-type all"
    queue = "normal"
    tasks = 1
elif hostname == "stampede2":
    if queue is None:
        queue = "skx-normal"  # SKX  ... the KNL nodes seem really slow
    #https://portal.tacc.utexas.edu/user-guides/stampede2#running

    print("preparing SLURM for stampede2...")
    host = HOST_STAMPEDE2 #defaulting to skx-normal

    python_cmd = "mpiexec.hydra -np 1 " + python_cmd

    if queue == "skx-normal":
        cores_per_node = 48
        if recover_mode:
            MAX_TIME_PER_TASK = 0.75  # in recover mode, can bit more agressive in timing (easier to continue if timeout)
        else:
            MAX_TIME_PER_TASK = 3.0  # MINUTES max

        if PYTHON_MAJOR_VERSION < 3:
            MAX_TASKS = 48 #point of seriously diminishing returns
            MAX_NODES = 3 #right now, pointless to go beyond 2 nodes
            MAX_TASKS_PER_NODE = 22 #actually, variable, encoded later
        else:
            MAX_TASKS = 4500
            MAX_NODES = 100
            MAX_TASKS_PER_NODE = 45 #still some memory issues, so back off the 48, just a bit
    else: #knl (much slower than SKX)
        cores_per_node = 68
        if recover_mode:
            MAX_TIME_PER_TASK = 5.0  # in recover mode, can bit more agressive in timing (easier to continue if timeout)
        else:
            MAX_TIME_PER_TASK = 6.0  # MINUTES max

        if PYTHON_MAJOR_VERSION < 3:
            MAX_TASKS = 48 #point of seriously diminishing returns
            MAX_NODES = 50 #right now, pointless to go beyond 2 nodes
            MAX_TASKS_PER_NODE = 6 #actually, variable, encoded later
        else:
            MAX_TASKS = 6800
            MAX_NODES = 100
            MAX_TASKS_PER_NODE = 68

    TIME_OVERHEAD = 1.0  # MINUTES of overhead to get started (per task call ... just a safety)

    time = "00:59:59"
    time_set = False
    email = "##SBATCH --mail-user\n##SBATCH --mail-type all"

    tasks = 1

elif hostname == "z50":
    host = HOST_LOCAL
    MAX_TASKS = 1 #
    TIME_OVERHEAD = 1.0  # MINUTES of overhead to get started (per task call ... just a safety)
    MAX_TIME_PER_TASK = 5.0  # MINUTES max, worst case expected time per task to execute (assumes minimal retries)
    cores_per_node = 1
    time = "00:59:59"
    time_set = False
    email = "##SBATCH --mail-user\n##SBATCH --mail-type all"
    queue = "normal"
    tasks = 1
else:
    print("Warning!! Preparing SLURM for generic host ...")
    MAX_TASKS = 1 #max allowed by TACC (for gpu or vis)
    cores_per_node = 1
    time = "00:59:59"
    time_set = False
    email = "##SBATCH --mail-user\n##SBATCH --mail-type all"
    queue = "gpu"
    tasks = 1








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


#check for tasks (optional)
i = -1
if "--tasks" in args:
    i = args.index("--tasks")
else:
    print("!!! --tasks NOT specified. Defaulting to (%d) task(s)." %tasks)

if i != -1:
    try:
        tasks = int(sys.argv[i + 1])
    except:
       print("Exception parsing --tasks")

    if tasks == 0:
        print("Auto set maximum tasks ...")
        if not time_set: #updated later on when we know the number of tasks per CPU
            time = "00:10:00" #10 minutes (no known elixer call takes more than 1 minute, give lots of slop
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
nodes = 1


dets_per_dispatch =  [] #list of counts ... the number of detection directories to list in the corresponding dispatch_xxx file
if tasks == 1:
    print("Only 1 task. Will not use dispatch.")

    run = python_cmd + path + ' ' + ' ' + ' '.join(sys.argv[1:]) + ' -f \n'
    dets_per_dispatch.append(1)
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
        print("Parsing directories to process. This may take a little while ...\n")

        if args.fcsdir is not None:
            subdirs = elixer.get_fcsdir_subdirs_to_process(args)
        else: #if (args.ra is not None):
            subdirs = elixer.get_hdf5_detectids_to_process(args)

        if tasks != 0:
            if tasks > len(subdirs):  # problem too many tasks requestd
                print("Error! Too many tasks (%d) requested. Only %d directories to process." % (tasks, len(subdirs)))
                exit(-1)
        else:
            tasks = min(MAX_TASKS,len(subdirs))

            if host == HOST_MAVERICK:
                #maverick forces 20 tasks per node, does not use --ntasks-per-node

                print("\nAdjusting tasks to nearest lower multiple of 20 ...")
                if tasks > 20:
                    tasks = tasks - tasks%20 #to deal with TACC parser issue

                nodes = tasks // cores_per_node
                if tasks % cores_per_node != 0:
                    nodes += 1

                ntasks_per_node = 20

            elif host == HOST_WRANGLER:
                #wrangler is a mess ... don't run more than 10 tasks and only on one node

                if PYTHON_MAJOR_VERSION < 3:
                    nodes = 1
                    ntasks_per_node = 10
                else:
                    if tasks < MAX_TASKS_PER_NODE:
                        nodes = 1
                    else:
                        nodes = min(tasks // MAX_TASKS_PER_NODE, MAX_NODES)
                        ntasks_per_node = tasks // nodes

            elif host == HOST_STAMPEDE2:
                #nominal, minium retries:
                #22 tasks per node up to 1 node   (22)
                #20 tasks per node up to 2 nodes  (40)
                #16 tasks per node up to 3 nodes  (48)

                pre_python_cmd = " export OMP_PROC_BIND=0 ; "

                if PYTHON_MAJOR_VERSION < 3:
                    if queue == "skx-normal":
                        if tasks <= 22:
                            ntasks_per_node = tasks
                            nodes = 1
                        elif tasks <= 40:
                            nodes = 2
                            ntasks_per_node = tasks // nodes + tasks % nodes
                        elif tasks <= 48: #point of seriously diminishing return
                            nodes = 3
                            ntasks_per_node = tasks // nodes + tasks % nodes
                        else: #cap at 48 (or 16 tasks per 3 nodes)
                            nodes = 3
                            ntasks_per_node = 16
                    else: #if queue == "normal":  # KNL
                        if tasks <= 6:
                            ntasks_per_node = tasks
                        else:
                            ntasks_per_node = 6

                        nodes = min(tasks // ntasks_per_node, MAX_NODES)
                else: #python 3 or better

                    if tasks < MAX_TASKS_PER_NODE:
                        nodes = 1
                    else:
                        nodes = min(tasks // MAX_TASKS_PER_NODE, MAX_NODES)
                        ntasks_per_node = tasks // nodes


            else:
                ntasks_per_node = tasks
                nodes = 1

            #fix the minimum (don't ask for more tasks per node than you have actual tasks to run)
            #only an issue if there is only one node requested
            ntasks_per_node = min(tasks,ntasks_per_node)

            print("%d detections as %d tasks on %d nodes at %d tasks-per-node" % (len(subdirs),tasks,nodes,ntasks_per_node))

        #dirs_per_file == how many detections (directory holding detection info) to add to each dispatch_xxx
        if tasks == 0:
            print("No tasks to execute. Exiting ...")
            exit(0)
        dirs_per_file = len(subdirs) // tasks  # int(floor(float(len(subdirs)) / float(tasks)))
        remainder = len(subdirs) % tasks
        dets_per_dispatch = np.full(tasks,dirs_per_file)
        dets_per_dispatch[0:remainder] += 1 #add one more per task to cover the remainder

        f = open("elixer.run", 'w')

        start_idx = 0
        for i in range(int(tasks)):
            fn = "dispatch_" + str(i).zfill(4)

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
            stop_idx = start_idx + dets_per_dispatch[i] #min(start_idx + dirs_per_file,len(subdirs))
            for j in range(start_idx,stop_idx):
                df.write(str(subdirs[j]) + "\n")

            df.close()

            start_idx = stop_idx #start where we left off

            #add  dispatch_xxx
            #run = "python " + path + ' ' + ' ' + ' '.join(sys.argv[1:]) + ' --dispatch ' + os.path.join(basename,fn) + ' -f \n'

            #may need to get rid of ra and dec

            #parms = remove_ra_dec(sys.argv[1:])
            #run = "cd " + fn + " ; python " + path + ' ' + ' ' + ' '.join(parms) + ' --dispatch ' + fn + ' -f ; cd .. \n'
            run = "cd " + fn + " ; " + pre_python_cmd + python_cmd + path + ' ' + ' ' + ' '.join(sys.argv[1:]) + ' --dispatch ' + fn + ' -f ; cd .. \n'
            f.write(run)

        f.close()
    except Exception as e:
        print(e)
        print("Error! Cannot create dispatch files.")

        exit(-1)


if not time_set: #update time
    try:
        mx = np.max(dets_per_dispatch)
        time = str(timedelta(minutes=TIME_OVERHEAD + MAX_TIME_PER_TASK * mx))
        print("--time %s" %time)

    except Exception as e:
        print("Error auto-setting time ... SLURM behavior may be unexpected.")
        print(e)  # for the repr
      #  print(str(e))  # for just the message
      #  print(e.args)  # the arguments that the exception has been called with.
        # the first one is usually the message.


#a little less efficient, but easier to read ... dump all the unncessary, human readable comments
#and just prep the machine used code

launch_str = None

if host == HOST_MAVERICK:
    slurm = "#!/bin/bash \n"
    slurm += "#SBATCH -J ELiXer              # Job name\n"
    slurm += "#SBATCH -n " + str(tasks) + "                  # Total number of tasks\n"
    slurm += "#SBATCH -N " + str(nodes) + "                  # Total number of nodes requested\n"
    slurm += "#SBATCH -p " + queue +"                 # Queue name\n"
    slurm += "#SBATCH -o ELIXER.o%j          # Name of stdout output file (%j expands to jobid)\n"
    slurm += "#SBATCH -t " + time + "            # Run time (hh:mm:ss)\n"
    slurm += "#SBATCH -A Hobby-Eberly-Telesco\n"
    slurm += email + "\n"
    slurm += "module unload xalt \n"
    slurm += "module load launcher\n"
    slurm += "export EXECUTABLE=$TACC_LAUNCHER_DIR/init_launcher\n"
    slurm += "export WORKDIR=. \n"
    slurm += "export CONTROL_FILE=elixer.run\n"
    slurm += "export TACC_LAUNCHER_NPHI=0\n"
    slurm += "export TACC_LAUNCHER_PHI_PPN=8\n"
    slurm += "export PHI_WORKDIR=.\n"
    slurm += "export PHI_CONTROL_FILE=phiparamlist\n"
    slurm += "export TACC_LAUNCHER_SCHED=interleaved\n"

    launch_str = "$TACC_LAUNCHER_DIR/paramrun SLURM $EXECUTABLE $WORKDIR $CONTROL_FILE $PHI_WORKDIR $PHI_CONTROL_FILE"


elif host == HOST_WRANGLER:

    slurm = "#!/bin/bash \n"
    slurm += "#SBATCH -J ELiXer              # Job name\n"
    #slurm += "#SBATCH -n " + str(tasks) + "                  # Total number of tasks\n"
    slurm += "#SBATCH -N " + str(nodes) + "                  # Total number of nodes requested\n"
    slurm += "#SBATCH --ntasks-per-node " + str(ntasks_per_node) + "       #Tasks per node\n"
    slurm += "#SBATCH -p " + queue + "                 # Queue name\n"
    slurm += "#SBATCH -o ELIXER.o%j          # Name of stdout output file (%j expands to jobid)\n"
    slurm += "#SBATCH -t " + time + "            # Run time (hh:mm:ss)\n"
    slurm += "#SBATCH -A Hobby-Eberly-Telesco\n"
    slurm += email + "\n"
    #slurm += "module unload xalt \n"
    slurm += "module load launcher\n"
    slurm += "export TACC_LAUNCHER_PPN=24\n"
    slurm += "export EXECUTABLE=$TACC_LAUNCHER_DIR/init_launcher\n"
    # note: wranger says WORKDIR and CONTROL_FILE deprecated, so replace with LAUNCHER_WORKDIR, LAUNCHER_JOB_FILE
    slurm += "export LAUNCHER_WORKDIR=$(pwd)\n"
    slurm += "export LAUNCHER_JOB_FILE=elixer.run\n"
    # just so the bottom part work as is and print w/o error
    slurm += "WORKDIR=$LAUNCHER_WORKDIR\n"
    slurm += "CONTROL_FILE=$LAUNCHER_JOB_FILE\n"

    slurm += "export LAUNCHER_SCHED=interleaved\n"

    launch_str = "$TACC_LAUNCHER_DIR/paramrun\n"


elif host == HOST_STAMPEDE2:

    slurm = "#!/bin/bash \n"
    slurm += "#SBATCH -J ELiXer              # Job name\n"
    #slurm += "#SBATCH -n " + str(tasks) + "                  # Total number of tasks\n"
    slurm += "#SBATCH -N " + str(nodes) + "                  # Total number of nodes requested\n"
    slurm += "#SBATCH --ntasks-per-node " + str(ntasks_per_node) + "       #Tasks per node\n"

    #if PYTHON_MAJOR_VERSION < 3:
    #    slurm += "#SBATCH --ntasks-per-node " + str(ntasks_per_node) + "       #Tasks per node\n"
    # else:
    #     slurm += "#SBATCH -n " + str(tasks) + "       #Total Tasks\n"

    slurm += "#SBATCH -p " + queue + "                 # Queue name\n"
    slurm += "#SBATCH -o ELIXER.o%j          # Name of stdout output file (%j expands to jobid)\n"
    slurm += "#SBATCH -t " + time + "            # Run time (hh:mm:ss)\n"
    slurm += "#SBATCH -A Hobby-Eberly-Telesco\n"
    slurm += email + "\n"

    #slurm += "module unload xalt \n"
    slurm += "module load launcher\n"
    slurm += "export LAUNCHER_PLUGIN_DIR=$TACC_LAUNCHER_DIR/plugins\n"
    slurm += "export LAUNCHER_RMI=SLURM\n"
    slurm += "export LAUNCHER_WORKDIR=$(pwd)\n"
    slurm += "export LAUNCHER_JOB_FILE=elixer.run\n"
    # just so the bottom part work as is and print w/o error
    slurm += "WORKDIR=$LAUNCHER_WORKDIR\n"
    slurm += "CONTROL_FILE=$LAUNCHER_JOB_FILE\n"

    slurm += "export LAUNCHER_SCHED=interleaved\n"


    launch_str = "$TACC_LAUNCHER_DIR/paramrun\n"

else:
    pass

#add the common logging/basic error checking to the end
slurm += "\
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
\n"

slurm += launch_str +"\n"

slurm += "echo \" \"\n"
slurm += "echo \" Parameteric Job Complete\"\n"
slurm += "echo \" \" "




if False: #old way
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
    #SBATCH -J ELiXer              # Job name\n\
    #SBATCH -n " + str(tasks) + "                  # Total number of tasks\n"

    slurm += "#SBATCH -N " + str(nodes) + "                  # Total number of nodes requested\n"
    slurm += "\
    #SBATCH -p " + queue +"                 # Queue name\n\
    #SBATCH -o ELIXER.o%j          # Name of stdout output file (%j expands to jobid)\n\
    #SBATCH -t " + time + "            # Run time (hh:mm:ss)\n\
    #SBATCH -A Hobby-Eberly-Telesco\n"
    slurm += email + "\n"
    slurm += "\
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
    #------------------General Options---------------------\n"

    if host == HOST_MAVERICK:
        slurm += "module unload xalt \n"
        slurm += "module load launcher\n"
        slurm += "export EXECUTABLE=$TACC_LAUNCHER_DIR/init_launcher\n"
        slurm += "export WORKDIR=.\n"
        slurm += "export CONTROL_FILE=elixer.run\n"

    elif host == HOST_WRANGLER:
        slurm += "module load launcher\n"
        slurm += "export TACC_LAUNCHER_PPN=24\n"
        slurm += "export EXECUTABLE=$TACC_LAUNCHER_DIR/init_launcher\n"
        #note: wranger says WORKDIR and CONTROL_FILE deprecated, so replace with LAUNCHER_WORKDIR, LAUNCHER_JOB_FILE
        slurm += "export LAUNCHER_WORKDIR=$(pwd)\n"
        slurm += "export LAUNCHER_JOB_FILE=elixer.run\n"
        #just so the bottom part work as is and print w/o error
        slurm += "WORKDIR=$LAUNCHER_WORKDIR\n"
        slurm += "CONTROL_FILE=$LAUNCHER_JOB_FILE\n"

    elif host == HOST_STAMPEDE2:
        slurm += "module load launcher\n"
        slurm += "export EXECUTABLE=$TACC_LAUNCHER_DIR/init_launcher\n"
        # note: wranger says WORKDIR and CONTROL_FILE deprecated, so replace with LAUNCHER_WORKDIR, LAUNCHER_JOB_FILE
        slurm += "export LAUNCHER_WORKDIR=$(pwd)\n"
        slurm += "export LAUNCHER_JOB_FILE=elixer.run\n"
        # just so the bottom part work as is and print w/o error
        slurm += "WORKDIR=$LAUNCHER_WORKDIR\n"
        slurm += "CONTROL_FILE=$LAUNCHER_JOB_FILE\n"
    else:
        slurm += "module load launcher\n"
        slurm += "export EXECUTABLE=$TACC_LAUNCHER_DIR/init_launcher\n"
        slurm += "export WORKDIR=. \n"
        slurm += "export CONTROL_FILE=elixer.run\n"

    slurm += "\
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
    echo \" Parameteric Job Complete \"\n \
    echo \" \" "
#end old way

try:
    f = open("elixer.slurm", 'w')
    f.write(slurm)
    f.close()
except:
    print("Error! Cannot create elixer.slurm")
    exit(-1)





#execute system command
print ("Calling SLURM queue ...")
if host == HOST_LOCAL:
    print("Here we will call: sbatch elixer.slurm")
else:
    os.system('sbatch elixer.slurm')
#





