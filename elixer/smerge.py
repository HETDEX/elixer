"""
SLURM merge
Basic merging of elixer catalog HDF5 files in a simple SLURM wrapper
only two generations of merge (assuming a maximum of 10,000 dispatch folders)
and ~40 tasks yields 250 files per task in first merge and then 40 in the second (and final)
Not the most efficient, but is fast to slap together and a big improvment over the linear merge
"""

#dont' bother with argparse ... we really only need the --dispatch <filename>

import glob
import os
import sys
import time
import numpy as np

try:
    from elixer import elixer_hdf5
except:
    import elixer_hdf5

def get_base_merge_files(topdir=".",pattern="dispatch_*/*/*_cat.h5",):
    """
    Scan for all _cat.h5 files to merge
    **DOES NOT INCLUDE elixer_merged_cat.h5 at the top directory if present**
    Return an unsorted list
    :param topdir: expected to be the direcotry containing all the dispatch_xxxx directories
    :return:
    """

    #try to avoid 'cache' by finding the basedir above me and then pattern = dispatch_*/<basedir_name>/*_cat.h5 ?
    #do not bother, all calls specify the pattern and do not hit the 'cache' direcotry for these usages

    files = []
    try:
        if pattern is None:
            reserved_name = os.path.join(topdir,"elixer_merged_cat.h5")
            files = sorted(glob.glob(os.path.join(topdir,"*_cat.h5")))
            files += sorted(glob.glob(os.path.join(topdir,"*_cat_*.h5")))
            files += sorted(glob.glob(os.path.join(topdir,"dispatch_*/*/*_cat.h5")))

            if reserved_name in files:
                #print(f"Excluding reserved file: {reserved_name}")
                files.remove(reserved_name)
                print(f"Reserved file found: {reserved_name}")
                print("Remove reserved file before re-running merge.")
                del files
                files = None
        else:
            files = glob.glob(os.path.join(topdir,pattern))
    except:
        pass

    return files


def merge_hdf5(fn_list=None,merge_fn="elixer_intermediate_merge.working",out_dir="."):
    """
    Similar to merge ... replaces merge ... joins ELiXer HDF5 catlogs.
    Does not check for duplicate entries.
    :param args:
    :return:
    """
    try:
        if len(fn_list) != 0:
            merge_fn = elixer_hdf5.merge_elixer_hdf5_files(merge_fn,fn_list)
            if merge_fn is not None:
                if os.path.basename(merge_fn) == "elixer_intermediate_merge.working":
                    os.rename(merge_fn,
                              os.path.join(out_dir,"elixer_intermediate_merge.h5"))
                    merge_fn =  os.path.join(out_dir,"elixer_intermediate_merge.h5")
                print("Done: " + merge_fn)
            else:
                print("Failed to write HDF5 catalog.")
        else:
            print("No HDF5 catalog files found. Are you in the directory with the dispatch_* subdirs?")
    except Exception as e:
        print(e)

def merge_unique(newfile,file1,file2):
    try:
        result = elixer_hdf5.merge_unique(newfile,file1,file2)

        if result:
            print("Merge Unique: Success. File = %s" %newfile)
        else:
            print("Merge Unique: FAIL.")
    except Exception as e:
        print(e)



def main():

    merge_list_fn = None
    merge_list = []
    args = list(map(str.lower, sys.argv))
    temp_wd = os.getcwd() #temporary working directory (if not set, use the cwd)

    i = -1
    if "--dispatch" in args:
        i = args.index("--dispatch")

    if i != -1:
        try:
            merge_list_fn = sys.argv[i + 1]
            print(f"merge_list_fn = {merge_list_fn}")
        except:
            print("Error! Cannot find mandatory parameter --dispatch")
            exit(-1)
    else:
        print("Error! Cannot find mandatory parameter --dispatch")
        exit(-1)

    if "--tmp" in args:
        i = args.index("--tmp")
        if i != -1:
            new_wd = sys.argv[i + 1]
            #there MUST be dispatch already and merge_list_fn is set
            #so, we create a subdir under the temp dir with the merge_list name (e.g. "merge_0000" or "final")
            if merge_list_fn != 'final':
                #merge_list_fn will be "merge_xxxx" and we want the directory to be "dispatch_xxxx"
                new_wd = os.path.join(new_wd,merge_list_fn.replace("merge","dispatch"))


            try:
                if os.access(new_wd, os.W_OK):
                    #os.chdir(new_wd)
                    temp_wd = new_wd
                elif os.path.exists(new_wd):
                    print(f"Warning! --tmp path is not writable: {new_wd}")
                    exit(-1)
                else: #try to create it
                    os.makedirs(new_wd,mode=0o755)
                    if os.access(new_wd, os.W_OK):
                        #os.chdir(new_wd) # do not actually change the working directory
                        temp_wd = new_wd
                    else:
                        print(f"Warning! --tmp path does not exist or is not writable: {new_wd}")
                        exit(-1)
            except:
                print("Exception processing command line for --tmp")
                exit(-1)


    #we have the list of files to merge
    #merge them linearly under a .working name
    #then change the name to .h5

    if merge_list_fn == 'final':
        #this is the top level final merge
        #find all previous generation merges files AND includ the elixer_merge.cat, if present
        #This runs from the top elixer output directory NOT from a dispatch_xxxx directory

        still_waiting = True
        print("Beginning final merge ... ")

        #how many do we expect?
        try:
            merge_count = len(get_base_merge_files(".","dispatch_*/merge_*"))
        except:
            merge_count = -1

        timeout_wait = 120.0 #seconds
        while still_waiting and timeout_wait > 0.0: #could sleep until the entire job times out if there is a problem
            #merge_list = get_base_merge_files(".","dispatch_*/*intermediate_merge.working")

            #assumes there are SOME .working on THIS node (the node that has the "final" merge)
            merge_list = get_base_merge_files(temp_wd, "dispatch_*/*intermediate_merge.working")
            if len(merge_list) > 0:
                print(f"Waiting on *working files to complete ...")
                time.sleep(10.0) #sleep 10 secs
                timeout_wait = 90.0 #seconds #reset to full timeout
            else:
                #get the final list FROM THE ORIGINAL cwd() as each sub-task copies back to there when done
                #  as we can be running on more than one node
                merge_list = get_base_merge_files(".", "dispatch_*/*intermediate_merge.h5")
                #merge_list = get_base_merge_files(temp_wd, "dispatch_*/*intermediate_merge.h5")
                if len(merge_list) < merge_count: #the final job could get kicked off before all the prior jobs are complete or even started
                    print(f"Waiting on merge_count to match. Current {len(merge_list)}, expected {merge_count}. Current timer: {timeout_wait:0.1f} ...")
                    time.sleep(10.0)             #so all the .working files might not even have been created yet
                    timeout_wait -= 10.0
                else:
                    still_waiting = False
                    print(f"Final Merge List {len(merge_list)}: {merge_list}")

        if still_waiting: #we timed out
            print(f"Timeout {timeout_wait:0.1f}s waiting on expected number {len(merge_list)}/{merge_count} of *intermediate_merge.h5 files. Aborting run.")
            dummy_list = [f"./dispatch_{str(n).zfill(4)}/elixer_intermediate_merge.h5" for n in range(merge_count)]
            missing = np.setdiff1d(dummy_list,merge_list)
            print(f"Missing files: {missing}")
            #YES, this writes to the original cwd not temp as we want to see this result always
            with open("elixer_merged_cat.fail","w") as f:
                f.write(f"Timeout {timeout_wait:0.1f}s waiting on expected number {len(merge_list)}/{merge_count} of *intermediate_merge.h5 files. Aborting run.\n")
                f.write(f"Missing files: {missing}\n")
                f.write(f"Expected files: {dummy_list}\n")
                f.write(f"Found files: {merge_list}\n")
            exit(-1)

        if len(merge_list) > 0:
            needs_unique = False

            if os.path.exists("elixer_merged_cat.h5"): #in the original cwd
                needs_unique = True

                #in temp dir
                merge_hdf5(merge_list, os.path.join(temp_wd,"elixer_merged_cat_new.h5"))

                #in original dir
                os.rename("elixer_merged_cat.h5","elixer_merged_cat_old.h5")


                merge_unique(os.path.join(temp_wd,"elixer_merged_cat.h5"),
                             "elixer_merged_cat_old.h5",os.path.join(temp_wd,"elixer_merged_cat_new.h5"))
                #merge_list.insert(0,"elixer_merged_cat.h5")
            else:
                merge_hdf5(merge_list, os.path.join(temp_wd,"elixer_merged_cat.h5"))



            #clean up
            print("Cleaning up ...")

            # elixer_merged_cat.h5 is in the temporary directory, need to copy back to the original cwd
            if os.getcwd() != temp_wd:
                import shutil
                #copy to cwd
                shutil.copy2(os.path.join(temp_wd,"elixer_merged_cat.h5"),"./elixer_merged_cat.h5")
                #remove the temp dir one
                os.remove(os.path.join(temp_wd, "elixer_merged_cat.h5"))

            for file in merge_list:
                if file != "elixer_merged_cat.h5":
                    print(f"Removing {file}")
                    os.remove(file)

            #temp wd
            if os.path.exists(os.path.join(temp_wd,"elixer_merged_cat_new.h5")):
                os.remove(os.path.join(temp_wd,"elixer_merged_cat_new.h5"))

            #original cwd
            if os.path.exists("elixer_merged_cat_old.h5"):
                os.remove("elixer_merged_cat_old.h5")

    else: #this is a generational merge

        #
        # NOTICE: we read from the original cwd (usually on /scratch) but merge to the temporary direcotry
        #

        #this could be a re-run
        #first check if intermediate.working exists ... if so, delete it and continue
        #if it does not exist, see if intermetediate.h5 exists, if so, this one is already done so skip

        do_merge = True
        try:
            if os.path.exists(os.path.join(temp_wd,"elixer_intermediate_merge.working")):
                os.remove(os.path.join(temp_wd,"elixer_intermediate_merge.working"))
                print(f"{merge_list_fn}: Incomplete. Delete working file and re-run.")
            elif not os.path.exists(os.path.join(temp_wd,"elixer_intermediate_merge.h5")):
                pass #continue to merge
            else: #this one is already done
                do_merge = False
                print(f"{merge_list_fn}: already done. Skipping.")
        except:
            print("Exception checking for recover conditions.")


        if do_merge:
            try:
                with open(merge_list_fn) as f:
                    for line in f:
                        line = line.rstrip("\n")
                        # based on being one level above dispatch_xxxx
                        if os.path.exists(line):
                            merge_list.append(line)
                        elif os.path.exists("../"+line):
                            merge_list.append("../" + line)

            except Exception as e:
                print("Could not open merge_list file")
                print(e)
                exit(-1)

            if len(merge_list)==0:
                print("No files to merge")
                exit(0)

            print(f"Merging {len(merge_list)} files, working from {os.getcwd()} ... ")
            print(merge_list)
            try: #merge_fn="elixer_intermediate_merge.working"
                merge_hdf5(merge_list,merge_fn=os.path.join(temp_wd,"elixer_intermediate_merge.working"),out_dir=temp_wd)

                #now copy from temp_wd to cwd() if we are using --tmp
                if temp_wd != os.getcwd():
                    import shutil
                    shutil.copy2(os.path.join(temp_wd, "elixer_intermediate_merge.h5"),
                                 os.path.join(os.getcwd(),"elixer_intermediate_merge.h5"))
                    os.remove(os.path.join(temp_wd, "elixer_intermediate_merge.h5"))

            except Exception as E:
                print(f"Exception. Merge failed from {os.getcwd()}")
                print(E)
            print("Intermediate merge complete")


if __name__ == '__main__':
    main()