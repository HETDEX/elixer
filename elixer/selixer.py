# wrapper for SLURM call to elixer
from __future__ import print_function
import sys
import os
import errno
try:
    from elixer import elixer
    from elixer import smerge
except:
    import elixer
    import smerge

import numpy as np
from math import ceil
from datetime import timedelta
import socket
from os import getenv

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

workbasedir = "/work"
try:
    workbasedir = getenv("WORK_BASEPATH")
    if workbasedir is None or len(workbasedir) == 0:
        workbasedir = "/work"
except:
    pass

hostname = socket.gethostname()
#print("+++++++++++++ put this BACK !!!!! ")
#hostname = "wrangler"

HOST_LOCAL = -1
HOST_UNKNOWN = 0
HOST_MAVERICK = 1
HOST_WRANGLER = 2
HOST_STAMPEDE2 = 3
HOST_LONESTAR6 = 4

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
# task == on the cluster ... generally == number of cores on that node (maybe artificially reduced to save memory)_
#         one line in the execution file [elixer.run]
#         becomes the number of dispatch_xxx subdirectories created
#         each will have one or more explicitly listed detections to process (each line in the dispatch_xxx file)
# core == single processing unit (runs one elixer call)
# node == group of cores (shares memory and other resources)


#form of loginxxx.<name>.tacc.utexas.edu for login nodes
#or cxxx-xxx.<name>.tacc.utexas.edu for compute nodes

args = list(map(str.lower,sys.argv)) #python3 map is no longer a list, so need to cast here

if ("--help" in args) or ("--version" in args):
    elixer.parse_commandline(auto_force=True)
    exit(0)


if "--cluster" in args:
    i = args.index("--cluster")
    if i != -1:
        try:
            from os.path import exists
            if exists(sys.argv[i + 1]):
                pass #all good
            else:
                print(f"Warning! --cluster file does not exist: {sys.argv[i + 1]}")
                exit(0)
        except:
            print("Exception processing command line for --cluster")
            exit(-1)




#check for --merge (if so just call elixer
MERGE = False
LOCAL_MERGE = False
if "--merge_local" in args:
    MERGE = True
    LOCAL_MERGE = True

if "--merge" in args:
    MERGE = True
    # if hostname == "z50":
    #     print("Testing SLURM merge ... remove z50 check")
    #     MERGE = True
    #     hostname = 'wrangler'
    # else:
    #     print("Calling ELiXer to merge catalogs and fiber files (ignoring all other parameters) ... ")
    #     elixer.merge()
    #     exit(0)



#used in the next two checks
dets = None
try:
    if "--dets" in args:# or "--hdr" in args:
        det_file_name = sys.argv[i + 1]
        if os.path.isfile(det_file_name):
            dets = np.loadtxt(det_file_name,dtype=int,usecols=0)#,max_rows=1)
        else:
            #could be a list
            dets = args.dets.replace(', ', ',').split(',')  # allow comma or comma-space separation
    else:
        dets = None
except:
    dets = None

#Continuum sanity check
#for a sanity check later:
continuum_mode = False
#if "--dets" in args:
    # i = args.index("--dets")
    # if i != -1:
if dets is not None:
    try:
        # det_file_name = sys.argv[i + 1]
        # if os.path.isfile(det_file_name):
        has_continuum_id = False

        #check the top and bottom for possible continuum objects
        #for simplicity, just read the whole column and read as int for size
        #dets = np.loadtxt(det_file_name,dtype=int,usecols=0)#,max_rows=1)

        #look for 3rd character as 9
        if str(dets[0])[2] == '9' or str(dets[-1])[2] == '9':
            has_continuum_id = True

        #del dets

        if "--continuum" in args:
            continuum_mode = True
        else:
            continuum_mode = False

        prompt = None
        if has_continuum_id and not continuum_mode:
            prompt = "Apparent continuum detectIDs in --dets file, but --continuum not specified. Continue anyway? (y/n)"
        elif not has_continuum_id and continuum_mode:
            prompt = "No apparent continuum detectIDs in --dets file, but --continuum IS specified. Continue anyway? (y/n)"
        #else all is okay

        if prompt is not None:
            r = input(prompt) #assumes Python3 or greater
            print()
            if len(r) > 0 and r.upper() !=  "Y":
                print ("Cancelled.\n")
                exit(0)
            else:
                print("Continuing ... \n")
    except Exception as e:
        pass

hdr_int = None
if "--hdr" in args:
    i = args.index("--hdr")
    if i != -1:
        try:
            hdr_int = int(sys.argv[i + 1])
        except:
            hdr_int = None


if hdr_int is not None and dets is not None:
    try:
        lead_char = np.unique([int(x[0]) for x in dets])
        if len(lead_char) != 1:
            #have mixed HDR Versions
            print(f"***** Error! Detections from different HDR versions is not allowed. Versions found: {lead_char}")
            exit(-1)
        elif len(lead_char) == 1:
            if lead_char[0] != hdr_int:
                print(f"***** Error! DetectID HDR version {lead_char[0]} does not match command line: --hdr {sys.argv[i + 1]}")
                exit(-1)
        else:
            #problem, can't be zero
            pass

    except:
        pass


if "--ooops" in args:
    ooops_mode = True
else:
    ooops_mode = False


if dets is not None:
    try:
        del dets
    except:
        pass

#--recover is now the default unless --no_recover is set
if "--no_recover" in args:
    recover_mode = False
else:
    recover_mode = True


if "--neighborhood" in args:
    i = args.index("--neighborhood")
    try:
        neighborhood = sys.argv[i + 1]
    except:
        neighborhood = 0
else:
    neighborhood = 0

if "--neighborhood_only" in args:
    neighborhood_only = True
else:
    neighborhood_only = False

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



#check for dependency (optional)
slurm_id_dependency = None
i = -1
if "--dependency" in args:
    i = args.index("--dependency")

if i != -1:
    try:
        slurm_id_dependency = sys.argv[i + 1]
    except:
       pass
else:
    pass



base_time_multiplier = 1.0
#gridsearch_task_boost = 0.0 #adds to the mx multiplier
if "--gridsearch" in args:
    try:
        i = args.index("--gridsearch")
        parms = sys.argv[i+1]
        try:
            gridsearch = parms.replace(')','')
            gridsearch = parms.replace('(','')
        except:
            pass

        gridsearch = tuple(map(float, gridsearch.split(',')))
        # technically there are 2x of these, but they do not add nearly as much time as a full elixer output
        # so we take the half-side and scale down by another 1/2 (so divide by 2 instead of multiply by 2)
        gridsearch_extracts = int(gridsearch[0]*2.0/gridsearch[1]+2)**2
        #gridsearch_task_boost = gridsearch_extracts * 0.15 / 60.0 #per gridsearch task, in minutes
        #also assumes only one shot per gridsearch
        base_time_multiplier = gridsearch_extracts * 0.0045

    except:
        base_time_multiplier = 3.0 #just to put something in
       # gridsearch_task_boost = None
    #just an average guess; the actual time depends on the grid width, cell size and number of shots

if "--mcmc" in args: #impacts the base_time_multiplier
    base_time_multiplier *= 4.0
    print("Warning! Force mcmc increases overall time by roughtly 4x")
#else:
#    force_mcmc = False

if ("--lyc" in args) or ("--deblend" in args): #some extra processing
    base_time_multiplier *= 2.0 #5x bump in time, largely due to deblending and extra fetching of spectra

if "--cluster" in args: #runs a clustering search per detectid but only re-runs a handful after that
    base_time_multiplier *= 0.2
    CLUSTER = True
else:
    CLUSTER = False

autoqueue_slurm = 1
if "--slurm" in args:
    i = args.index("--slurm")
    try:
        autoqueue_slurm = int(sys.argv[i + 1])
    except:
        autoqueue_slurm = 1


if MERGE:
    base_time_multiplier = 0.05

if "tacc.utexas.edu" in hostname:
    hostname = hostname.split(".")[1]

FILL_CPU_TASKS = 10 #don't add another node until each CPU on the current node(s) hit this number
                    #e.g. roughly the number of detections per CPU (or row in the run.h5 filer),
                    # or the detectios per dispatch_xxxx (generally +/- 1 detection, but at low number of nodes can be
                    # up to 50% more ... so if the fill tasks is 10, could get 15 if only a few nodes are in use)
                    #this is changed based on the server and CPU type below
MAX_DETECTS_PER_CPU = 9999999 #do not execute this job of the dispatch_xxxx list count exceeds this value
MAX_TASKS_PER_NODE =1 #default (local machine)
MAX_NODES=1


#note MAX_TASKS is the maximum number of dispatchs to create
#     MAX_TASKS_PER_NODE is effectively the number of CORES to use per node

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
    tasks = 0
elif hostname == "wrangler":
    #!!! right now, for whatever reason, memmory is a problem for wrangler
    #and it can handle only 4 tasks per node (maximum)
    #It takes about 1 minute to run a task (lets call it 90 seconds to be safe)
    #set -N and -n s|t n/N <= 4
    print("preparing SLURM for wrangler...")
    MAX_DETECTS_PER_CPU = 25
    host = HOST_WRANGLER
    MAX_TASKS = 10
    MAX_NODES = 1
    #MAX_TASKS_PER_NODE = 10 #actually, variable, encoded later
    TIME_OVERHEAD = 2.0  # MINUTES of overhead to get started (per task call ... just a safety)
    if recover_mode:
        if neighborhood_only:
            MAX_TIME_PER_TASK = 0.25
        else:
            MAX_TIME_PER_TASK = 1.0 #in recover mode, can bit more agressive in timing (easier to continue if timeout)
    else:
        if neighborhood_only:
            MAX_TIME_PER_TASK = 0.5
        else:
            MAX_TIME_PER_TASK = 5.0  # MINUTES max, worst case expected time per task to execute (assumes minimal retries)

    cores_per_node = 24

    if PYTHON_MAJOR_VERSION < 3:
        MAX_TASKS = 216  # point of seriously diminishing returns
        MAX_NODES = 36  # right now, pointless to go beyond 2 nodes
        MAX_TASKS_PER_NODE = 6  # actually, variable, encoded later
    else:
        MAX_DETECTS_PER_CPU = 50
        if MERGE:
            MAX_TASKS = 100
            MAX_NODES = 4
            MAX_TASKS_PER_NODE = 24
        else:
            MAX_TASKS = 10000  # 20*36=720, so 720 in one pass; as "dispatch" or line in .run file finishes, the next is picked up
            MAX_NODES = 32
            MAX_TASKS_PER_NODE = 20 #need extra memory (128GB/20 instead of 128GB/24)

    time = "00:59:59"
    time_set = False
    email = "##SBATCH --mail-user\n##SBATCH --mail-type all"
    if queue is None:
        queue = "normal"
    tasks = 0
elif hostname == "stampede2":
    if queue is None:
        queue = "skx-normal"  # SKX  ... the KNL nodes seem really slow
    #https://portal.tacc.utexas.edu/user-guides/stampede2#running

    print("preparing SLURM for stampede2...")
    host = HOST_STAMPEDE2 #defaulting to skx-normal

    python_cmd = "mpiexec.hydra -np 1 " + python_cmd

    if queue == "skx-normal": #(192GB per node)
        MAX_DETECTS_PER_CPU = 100
        cores_per_node = 48
        if recover_mode:
            if neighborhood_only:
                MAX_TIME_PER_TASK = 0.25
            elif neighborhood == 0:
                MAX_TIME_PER_TASK = 1.0 #0.9 ver 1.18+ with hetdex_api is slower than HDF5 direct
            else:
                MAX_TIME_PER_TASK = 1.3 #1.2  # in recover mode, can bit more agressive in timing (easier to continue if timeout)
        else:
            if neighborhood_only:
                MAX_TIME_PER_TASK = 0.5
            elif neighborhood == 0:
                MAX_TIME_PER_TASK = 2.0
            else:
                MAX_TIME_PER_TASK = 3.0  # MINUTES max

        if PYTHON_MAJOR_VERSION < 3:
            MAX_TASKS = 48 #point of seriously diminishing returns
            MAX_NODES = 3 #right now, pointless to go beyond 2 nodes
            MAX_TASKS_PER_NODE = 22 #actually, variable, encoded later
        else:
            FILL_CPU_TASKS = 30
            if MERGE:
                MAX_TASKS = 100
                MAX_NODES = 2
                MAX_TASKS_PER_NODE = 48
            else:
                MAX_TASKS = 10000
                MAX_NODES = 128 #why 20?
                MAX_TASKS_PER_NODE = 48 #still some memory issues ... this gives us a little more room
            #MAX_TASKS = MAX_NODES * MAX_TASKS_PER_NODE #800
    else: #knl (much slower than SKX and much less memory (96 GB per node)
        cores_per_node = 68
        if recover_mode:
            if neighborhood_only:
                MAX_TIME_PER_TASK = 0.5
            elif neighborhood == 0:
                MAX_TIME_PER_TASK = 4.0
            else:
                MAX_TIME_PER_TASK = 5.0  # in recover mode, can bit more agressive in timing (easier to continue if timeout)
        else:
            if neighborhood_only:
                MAX_TIME_PER_TASK = 1.5
            elif neighborhood == 0:
                MAX_TIME_PER_TASK = 4.5
            else:
                MAX_TIME_PER_TASK = 6.0  # MINUTES max

        if PYTHON_MAJOR_VERSION < 3:
            MAX_TASKS = 48 #point of seriously diminishing returns
            MAX_NODES = 50 #right now, pointless to go beyond 2 nodes
            MAX_TASKS_PER_NODE = 6 #actually, variable, encoded later
        else:

            if MERGE:
                MAX_TASKS = 100
                MAX_NODES = 2
                MAX_TASKS_PER_NODE = 48 #memory can be a problem, even for merge
            else:
                MAX_TASKS = 10000
                MAX_NODES = 100
                MAX_TASKS_PER_NODE = 20
            #MAX_TASKS = MAX_NODES * MAX_TASKS_PER_NODE  # 2000

    TIME_OVERHEAD = 4.0  # MINUTES of overhead to get started (per task call ... just a safety)

    time = "00:59:59"
    time_set = False
    email = "##SBATCH --mail-user\n##SBATCH --mail-type all"

    tasks = 0
elif hostname == "lonestar6" or hostname == 'ls6':
    if queue is None:
        queue = "normal"  # SKX  ... the KNL nodes seem really slow
    # https://docs.tacc.utexas.edu/hpc/lonestar6/
    # standard compute node: num =  560 each with 128 cores, 256 GB RAM and 144GB disk on /tmp
    # vm-small
    # GPU nodes
    # /scratch purge after minimum of 10days
    # QUEUES:
       #
       #        Name       MinNode       MaxNode     MaxWall     MaxNodePU MaxJobsPU MaxSubmit
       #    gpu-a100                          16  2-00:00:00            32         8        40
       #       large            65           256  2-00:00:00           256         1        20
       #      normal             1            64  2-00:00:00           128        20       200
       #       debug                         576  2-00:00:00           576        30        60
       # development                           4    02:00:00             6         1         3

    print("preparing SLURM for lonestar6 ...")
    host = HOST_LONESTAR6  # defaulting to skx-normal

    #lonestar6 does not use ibrun or mpiexec here
    #python_cmd = "ibrun -np 1 " + python_cmd

    if queue == "normal":  # (192GB per node)
        MAX_DETECTS_PER_CPU = 100
        cores_per_node = 128
        if recover_mode:
            if neighborhood_only:
                MAX_TIME_PER_TASK = 0.25
            elif neighborhood == 0:
                MAX_TIME_PER_TASK = 1.0  # 0.9 ver 1.18+ with hetdex_api is slower than HDF5 direct
            else:
                MAX_TIME_PER_TASK = 1.3  # 1.2  # in recover mode, can bit more agressive in timing (easier to continue if timeout)
        else:
            if neighborhood_only:
                MAX_TIME_PER_TASK = 0.5
            elif neighborhood == 0:
                MAX_TIME_PER_TASK = 2.0
            else:
                MAX_TIME_PER_TASK = 3.0  # MINUTES max

        if PYTHON_MAJOR_VERSION < 3:
            print("Python < 3 No longer supported")
            exit(-1)
        else:
            FILL_CPU_TASKS = 15
            if MERGE: #use same tasks limit as full run
                MAX_TASKS = 128
                MAX_NODES = 2
                MAX_TASKS_PER_NODE = 64  #need usually around 4-4.5GB per core, 256GB/ (4GB/task) = 64 tasks, 4.5GB = 58
            else:
                MAX_TASKS = 10000
                MAX_NODES = 128
                if neighborhood == 0:
                    MAX_TASKS_PER_NODE = 64  # need usually around 4GB per core, 256GB/ (4GB/task) = 64 tasks, 4.5GB = 56
                else:
                    MAX_TASKS_PER_NODE = 56  # need usually around 4GB per core, 256GB/ (4GB/task) = 64 tasks, 4.5GB = 56
            # MAX_TASKS = MAX_NODES * MAX_TASKS_PER_NODE #800
    elif queue == 'vm-small':  #much smaller, less memory, just a guess at this time
        cores_per_node = 16
        if recover_mode:
            if neighborhood_only:
                MAX_TIME_PER_TASK = 0.5
            elif neighborhood == 0:
                MAX_TIME_PER_TASK = 4.0
            else:
                MAX_TIME_PER_TASK = 5.0  # in recover mode, can bit more agressive in timing (easier to continue if timeout)
        else:
            if neighborhood_only:
                MAX_TIME_PER_TASK = 1.5
            elif neighborhood == 0:
                MAX_TIME_PER_TASK = 4.5
            else:
                MAX_TIME_PER_TASK = 6.0  # MINUTES max

        if PYTHON_MAJOR_VERSION < 3:
            print("Python < 3 No longer supported")
            exit(-1)
        else:
            if MERGE:
                MAX_TASKS = 100
                MAX_NODES = 2
                MAX_TASKS_PER_NODE = 50  # memory can be a problem, even for merge
            else:
                MAX_TASKS = 10000
                MAX_NODES = 100
                MAX_TASKS_PER_NODE = 2
            # MAX_TASKS = MAX_NODES * MAX_TASKS_PER_NODE  # 2000
    TIME_OVERHEAD = 4.0  # MINUTES of overhead to get started (per task call ... just a safety)

    time = "00:59:59"
    time_set = False
    email = "##SBATCH --mail-user\n##SBATCH --mail-type all"

    tasks = 0

elif hostname in ["z50","dg5"]:
    host = HOST_LOCAL
    MAX_TASKS = 100 # #dummy value just for testing
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
elif not MERGE:
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

timex = 1.0
if "--timex" in args:
    i = args.index("--timex")
    try:
        timex = float(sys.argv[i + 1])
    except:
        timex = 1.0
else:
    timex = 1.0

#if nophot is specified (do not use photometric imaging, so will not retrieve it)
#we want to cut the nominal time in half, so, it is the equivalent of cuting timex down
if "--nophoto" in args:
    timex *= 0.7  # 0.5 is a bit too much ...

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



#check for ntasks_per_node (optional)
i = -1
ntasks_per_node = 0
if "--ntasks_per_node" in args:
    i = args.index("--ntasks_per_node")
else:
    print(f"--ntasks_per_node NOT specified. Defaulting to {hostname} {MAX_TASKS_PER_NODE}")

if i != -1:
    try:
        ntasks_per_node = int(sys.argv[i + 1])
    except:
       print("Exception parsing --ntasks_per_node")

    if ntasks_per_node == 0:
        print("Auto set maximum ntasks_per_node ...")
        if not time_set: #updated later on when we know the number of tasks per CPU
            time = "00:10:00" #10 minutes (no known elixer call takes more than 1 minute, give lots of slop
    elif (ntasks_per_node < 1) or (ntasks_per_node > MAX_TASKS_PER_NODE):
        print (f"Invalid --ntasks_per_node value. Must be 0 (auto-max) or between 1 to {MAX_TASKS_PER_NODE} inclusive.")
        exit(-1)
    else:
        MAX_TASKS_PER_NODE = ntasks_per_node
else:
    pass

#check for nodes
max_nodes = 0
i = -1
if "--nodes" in args:
    i = args.index("--nodes")

if i != -1:
    try:
        max_nodes = int(sys.argv[i + 1])
    except:
       print("Exception parsing --nodes")

    if max_nodes == 0:
        print(f"--nodes (0). Auto set maximum nodes {hostname}:{MAX_NODES} ...")
    elif (max_nodes < 1) or (max_nodes > MAX_NODES):
        print ("Invalid --nodes value.", max_nodes)
        exit(-1)
    else: #max_nodes is valid, so set it as the new limit
        MAX_NODES = max_nodes
        print("Redfinining MAX_NODES to ", max_nodes)
else:
    print(f"--nodes not specified. Auto set maximum nodes {hostname}:{MAX_NODES} ...")


if not MERGE:
    if not os.path.isdir(basename):
        try:
            os.makedirs(basename)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                print("Error! Cannot create output directory: %s" % basename)
                exit(-1)

    os.chdir(basename)

### elixer.run
if not MERGE:
    path = os.path.join(os.path.dirname(sys.argv[0]),"elixer.py")
else:
    path = os.path.join(os.path.dirname(sys.argv[0]),"smerge.py")
nodes = 1
if ntasks_per_node < 1:
    ntasks_per_node = 1

dets_per_dispatch =  [] #list of counts ... the number of detection directories to list in the corresponding dispatch_xxx file
if tasks == 1:
    print("Only 1 task. Will not use dispatch.")
    ntasks_per_node = 1

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

        subdirs = [] #this is a list of detectids OR RA and Decs (the length of the list is used and each "row" is written out later)

        if MERGE:
            if LOCAL_MERGE:
                subdirs = smerge.get_base_merge_files(".",pattern=None)
                if subdirs is None or len(subdirs)==0:
                    print("No files to merge. Exiting")
                    exit(0)
            else:
                subdirs = smerge.get_base_merge_files()
        else:
            if args.fcsdir is not None:
                subdirs = elixer.get_fcsdir_subdirs_to_process(args)
            # elif args.aperture is not None:
            #     #need to work on this here
            #     print("SLURM/dispatch (re)extraction not yet ready....")
            #     exit(0)
            else: #if (args.ra is not None):
                #either a --dets list or a --coords list, but either way, get a list of HETDEX detectids to process
                subdirs = elixer.get_hdf5_detectids_to_process(args,as_rows=True)#just want the number of rows

        if tasks != 0:
            if tasks > len(subdirs):  # problem too many tasks requestd
                print("Error! Too many tasks (%d) requested. Only %d directories to process." % (tasks, len(subdirs)))
                exit(-1)
        else:
            if MERGE:
                tasks = min(MAX_TASKS, len(subdirs)//2)
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

                    target_nodes = max(1, int(len(subdirs) / (FILL_CPU_TASKS * MAX_TASKS_PER_NODE)))
                    target_tasks = min(tasks,target_nodes * MAX_TASKS_PER_NODE)

                    nodes = min(target_nodes, MAX_NODES)
                    ntasks_per_node = min(target_tasks, MAX_TASKS_PER_NODE)
                    tasks = min(target_tasks, MAX_TASKS)

            elif host == HOST_STAMPEDE2:
                #nominal, minium retries:
                #22 tasks per node up to 1 node   (22)
                #20 tasks per node up to 2 nodes  (40)
                #16 tasks per node up to 3 nodes  (48)

                #todo: now seeing an error reported from stampede2 20190601 that the value is invalid
                #todo: so, turning this off again
                #pre_python_cmd = " export OMP_PROC_BIND=0 ; "

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
                        ntasks_per_node = tasks
                    else:
                        #nodes = min(tasks // MAX_TASKS_PER_NODE, MAX_NODES)
                        #ntasks_per_node = tasks // nodes

                        target_nodes = max(1,int(len(subdirs)/(FILL_CPU_TASKS * MAX_TASKS_PER_NODE)))
                        target_tasks = min(tasks,target_nodes * MAX_TASKS_PER_NODE) #target_nodes * MAX_TASKS_PER_NODE

                        nodes = min(target_nodes,MAX_NODES)
                        ntasks_per_node = min(target_tasks,MAX_TASKS_PER_NODE)
                        tasks = min(target_tasks,MAX_TASKS)
            elif host == HOST_LONESTAR6:

                if PYTHON_MAJOR_VERSION < 3:
                    print("Python < 3 not supported.")
                    exit(-1)
                else: #python 3 or better

                    if tasks < MAX_TASKS_PER_NODE:
                        nodes = 1
                        ntasks_per_node = tasks
                    else:
                        #nodes = min(tasks // MAX_TASKS_PER_NODE, MAX_NODES)
                        #ntasks_per_node = tasks // nodes

                        target_nodes = max(1,round(len(subdirs)/(FILL_CPU_TASKS * MAX_TASKS_PER_NODE)))
                        target_dispatches = min(tasks,target_nodes * MAX_TASKS_PER_NODE) #target_nodes * MAX_TASKS_PER_NODE
                        #e.g. the target number of dispatches

                        nodes = min(target_nodes,MAX_NODES)
                        ntasks_per_node = min(target_dispatches,MAX_TASKS_PER_NODE) #so we don't duplicate if too few
                        tasks = min(target_dispatches,MAX_TASKS) #actual number of dispatches to make

            else:
                ntasks_per_node = tasks
                nodes = 1

            #fix the minimum (don't ask for more tasks per node than you have actual tasks to run)
            #only an issue if there is only one node requested
            ntasks_per_node = min(tasks,ntasks_per_node)
            if not MERGE:
                print("%d detections as %d tasks (dispatch_xxxx) on %d nodes at ~ %d tasks-per-node" %
                      (len(subdirs),tasks,nodes,ntasks_per_node))
            else:
                print("%d catalogs to merge as %d tasks (dispatch_xxxx) on %d nodes at ~ %d tasks-per-node" %
                      (len(subdirs),tasks,nodes,ntasks_per_node))


        #dirs_per_file == how many detections (directory holding detection info) to add to each dispatch_xxx
        if tasks == 0:
            print("No tasks to execute. Exiting ...")
            exit(0)
        dirs_per_file = len(subdirs) // tasks  # int(floor(float(len(subdirs)) / float(tasks)))

        if not (MERGE or CLUSTER) and (dirs_per_file > MAX_DETECTS_PER_CPU):
            print("Maximum allowed CPU loading exceeded. Each CPU set to process %d detections." % dirs_per_file)
            print("The maximum configured limit is %d" % MAX_DETECTS_PER_CPU)
            print("Reduce the number of detections input and try again.")
            exit(0)


        remainder = len(subdirs) % tasks
        dets_per_dispatch = np.full(tasks,dirs_per_file)
        dets_per_dispatch[0:remainder] += 1 #add one more per task to cover the remainder

        if not MERGE:
            f = open("elixer.run", 'w')
        else:
            f = open("elixer_merge.run", 'w')

        start_idx = 0
        for i in range(int(tasks)):

            fn = "dispatch_" + str(i).zfill(4)
            if MERGE:
                fn2 = "merge_" + str(i).zfill(4)
            else:
                fn2 = fn

            if not os.path.isdir(fn):
                try:
                    os.makedirs(fn)
                except OSError as exception:
                    if exception.errno != errno.EEXIST:
                        print ("Fatal. Cannot create output directory: %s" % fn)
                        exit(-1)

            df = open(os.path.join(fn,fn2), 'w')
            #content = ""

            #start_idx = i * dirs_per_file
            stop_idx = start_idx + dets_per_dispatch[i] #min(start_idx + dirs_per_file,len(subdirs))
            if isinstance(subdirs[start_idx],np.ndarray):
                for j in range(start_idx,stop_idx):
                    df.write(" ".join(subdirs[j].astype(str)) + "\n") #space separated
            else:
                for j in range(start_idx,stop_idx):
                    df.write(str(subdirs[j]) + "\n")

            df.close()

            start_idx = stop_idx #start where we left off

            #add  dispatch_xxx
            #run = "python " + path + ' ' + ' ' + ' '.join(sys.argv[1:]) + ' --dispatch ' + os.path.join(basename,fn) + ' -f \n'

            #may need to get rid of ra and dec

            #parms = remove_ra_dec(sys.argv[1:])
            #run = "cd " + fn + " ; python " + path + ' ' + ' ' + ' '.join(parms) + ' --dispatch ' + fn + ' -f ; cd .. \n'
            if not MERGE:
                run = "cd " + fn + " ; " + pre_python_cmd + python_cmd + path + ' ' + ' ' + ' '.join(sys.argv[1:]) + ' --dispatch ' + fn + ' -f ; cd .. \n'
            else:
                run = "cd " + fn + " ; " + pre_python_cmd + python_cmd + path + ' ' + ' ' + \
                      ' '.join(sys.argv[1:]) + ' --dispatch ' + fn2 + ' -f ; cd .. \n'

            f.write(run)

        if MERGE:
            #need to add one more task line for the top-level final merge
            #this runs from the top (no directory change)
            run = pre_python_cmd + python_cmd + path + ' ' + ' ' + \
                  ' '.join(sys.argv[1:]) + ' --dispatch final' + ' -f \n'
            f.write(run)

        f.close()
    except Exception as e:
        print(e)
        print("Error! Cannot create dispatch files.")

        exit(-1)


if not time_set: #update time
    try:
        mx = np.max(dets_per_dispatch)

        try:
            #number of nodes could be restricted so we are overloaded s|t a given core on a node may cycle back
            #and pick up another task to run
            mult = tasks / (nodes * ntasks_per_node)
        except:
            mult = 1.0
        #
        # if gridsearch_task_boost is not None:
        #     mx += gridsearch_task_boost

        # set a minimum time ... always AT LEAST 5 or 10 minutes requested?
        minutes = int(TIME_OVERHEAD + MAX_TIME_PER_TASK * mx * mult * base_time_multiplier * timex)

        if continuum_mode:
            minutes = int(minutes * 1.05) #small boost since continuum objects have extra processing
        time = str(timedelta(minutes=max(minutes,10.0)))
        print(f"auto-set time: TIME_OVERHEAD {TIME_OVERHEAD} + MAX_TIME_PER_TASK {MAX_TIME_PER_TASK} x mx {mx} "
              f"x mult {mult} x base_time_multiplier {base_time_multiplier} x timex {timex}")
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
    if MERGE:
        slurm += "export CONTROL_FILE=elixer_merge.run\n"
    else:
        slurm += "export CONTROL_FILE=elixer.run\n"
#    slurm += "export CONTROL_FILE=elixer.run\n"
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

    if ooops_mode:
        # slurm += "module use /work/01255/siliu/stampede2/ooops/modulefiles/ \n"
        # slurm += "ml ooops/1.0\n"
        # slurm += "set_io_param 1 low\n"

        #updated 01-23-2020
        # slurm += "module use /work/01255/siliu/launcher/modulefiles/ \n"
        # slurm += "module load launcher/3.5 \n"
        # slurm += "module use /work/01255/siliu/stampede2/ooops/modulefiles/ \n"
        # slurm += "module load ooops/1.0 \n"
        # slurm += "set_io_param 1 45.00, 35.00, 20.00, 50.00 \n"

        # updated 02-12-2020 to match stamped2
        slurm += "module use " + workbasedir + "/01255/siliu/stampede2/ooops/modulefiles/ \n"
        slurm += "module load ooops \n" #"/1.0 \n"
        slurm += "export IO_LIMIT_CONFIG=" + workbasedir + "/01255/siliu/stampede2/ooops/1.0/conf/config_low \n"
        slurm += "set_io_param 0 low\n"

    #slurm += "module unload xalt \n"
    slurm += "module load launcher\n"
    slurm += "export TACC_LAUNCHER_PPN=24\n"
    slurm += "export EXECUTABLE=$TACC_LAUNCHER_DIR/init_launcher\n"
    # note: wranger says WORKDIR and CONTROL_FILE deprecated, so replace with LAUNCHER_WORKDIR, LAUNCHER_JOB_FILE
    slurm += "export LAUNCHER_WORKDIR=$(pwd)\n"
    if MERGE:
        slurm += "export LAUNCHER_JOB_FILE=elixer_merge.run\n"
    else:
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

    if ooops_mode:
        # slurm += "module use /work/01255/siliu/stampede2/ooops/modulefiles/ \n"
        # slurm += "ml ooops/1.0\n"
        # slurm += "set_io_param 1\n"

        #updated 01-23-2020
        slurm += "module use " + workbasedir + "/01255/siliu/stampede2/ooops/modulefiles/ \n"
        slurm += "module load ooops \n" #"/1.0 \n"
        slurm += "export IO_LIMIT_CONFIG=" + workbasedir + "/01255/siliu/stampede2/ooops/1.0/conf/config_low \n"
        slurm += "set_io_param 0 low\n"
        #slurm += "/work/01255/siliu/stampede2/ooops/1.0/bin/set_io_param 0 low \n"

    #slurm += "module unload xalt \n"
    slurm += "module load launcher\n"
    slurm += "export LAUNCHER_PLUGIN_DIR=$TACC_LAUNCHER_DIR/plugins\n"
    slurm += "export LAUNCHER_RMI=SLURM\n"
    slurm += "export LAUNCHER_WORKDIR=$(pwd)\n"
    if MERGE:
        slurm += "export LAUNCHER_JOB_FILE=elixer_merge.run\n"
    else:
        slurm += "export LAUNCHER_JOB_FILE=elixer.run\n"
    #slurm += "export LAUNCHER_JOB_FILE=elixer.run\n"
    # just so the bottom part work as is and print w/o error
    slurm += "WORKDIR=$LAUNCHER_WORKDIR\n"
    slurm += "CONTROL_FILE=$LAUNCHER_JOB_FILE\n"

    slurm += "export LAUNCHER_SCHED=interleaved\n"
    slurm += "export LD_PRELOAD=" + workbasedir + "/00410/huang/share/patch/myopen.so \n"

    launch_str = "$TACC_LAUNCHER_DIR/paramrun\n"

elif host == HOST_LONESTAR6:

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
    slurm += "#SBATCH -e ELIXER.e%j          # Name of stderr output file (%j expands to jobid)\n"
    slurm += "#SBATCH -t " + time + "            # Run time (hh:mm:ss)\n"
    slurm += "#SBATCH -A AST23008\n"
    slurm += email + "\n"

    #assume ooops not for LoneStar6
    # if ooops_mode:
    #
    #     #updated 01-23-2020
    #     slurm += "module use " + workbasedir + "/01255/siliu/stampede2/ooops/modulefiles/ \n"
    #     slurm += "module load ooops \n" #"/1.0 \n"
    #     slurm += "export IO_LIMIT_CONFIG=" + workbasedir + "/01255/siliu/stampede2/ooops/1.0/conf/config_low \n"
    #     slurm += "set_io_param 0 low\n"
    #     #slurm += "/work/01255/siliu/stampede2/ooops/1.0/bin/set_io_param 0 low \n"

    #slurm += "module unload xalt \n"
    slurm += "module load launcher\n"
    slurm += "export LAUNCHER_PLUGIN_DIR=$TACC_LAUNCHER_DIR/plugins\n"
    slurm += "export LAUNCHER_RMI=SLURM\n"
    slurm += "export LAUNCHER_WORKDIR=$(pwd)\n"
    if MERGE:
        slurm += "export LAUNCHER_JOB_FILE=elixer_merge.run\n"
    else:
        slurm += "export LAUNCHER_JOB_FILE=elixer.run\n"
    #slurm += "export LAUNCHER_JOB_FILE=elixer.run\n"
    # just so the bottom part work as is and print w/o error
    slurm += "WORKDIR=$LAUNCHER_WORKDIR\n"
    slurm += "CONTROL_FILE=$LAUNCHER_JOB_FILE\n"

    slurm += "export LAUNCHER_SCHED=interleaved\n"


    launch_str = "$TACC_LAUNCHER_DIR/paramrun\n"
elif host == HOST_LOCAL:
    slurm = "HOST_LOCAL ignored"
    launch_str = "nothing to launch"

#added per https://portal.tacc.utexas.edu/tutorials/managingio#ooops
#slurm += "export LD_PRELOAD=" + workbasedir +"/00410/huang/share/patch/myopen.so \n"

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

try:
    if MERGE:
        f = open("elixer_merge.slurm", 'w')
    else:
        f = open("elixer.slurm", 'w')
    f.write(slurm)
    f.close()
except:
    print("Error! Cannot create SLURM file")
    exit(-1)


############################
#execute system command
############################
print ("Calling SLURM queue ...")
if host == HOST_LOCAL:
    print("Here we will call: sbatch elixer.slurm")
elif autoqueue_slurm == 0:
    print("SLURM job created, but not auto-queued.")
else:
    if MERGE:
        os.system('sbatch elixer_merge.slurm')
    elif slurm_id_dependency is not None:
        if "after" in slurm_id_dependency:
            os.system(f'sbatch --dependency {slurm_id_dependency} elixer.slurm')
        else:
            os.system(f'sbatch --dependency afterany:{slurm_id_dependency} elixer.slurm')
    else:
        os.system('sbatch elixer.slurm')
