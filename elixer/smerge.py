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

def get_base_merge_files(topdir=".",pattern="dispatch_*/*/*_cat.h5"):
    """
    Scan for all _cat.h5 files to merge
    **DOES NOT INCLUDE elixer_merged_cat.h5 at the top directory if present**
    Return an unsorted list
    :param topdir: expected to be the direcotry containing all the dispatch_xxxx directories
    :return:
    """

    files = []
    try:
        files = glob.glob(os.path.join(topdir,pattern))
    except:
        pass

    return files


def merge_hdf5(fn_list=None,merge_fn="elixer_intermediate_merge.working"):
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
                if merge_fn == "elixer_intermediate_merge.working":
                    os.rename("elixer_intermediate_merge.working","elixer_intermediate_merge.h5")
                    merge_fn = "elixer_intermediate_merge.h5"
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

    #we have the list of files to merge
    #merge them linearly under a .working name
    #then change the name to .h5

    if merge_list_fn == 'final':
        #this is the top level final merge
        #find all previous generation merges files AND includ the elixer_merge.cat, if present
        still_waiting = True
        print("Beginning final merge ... ")

        #how many do we expect?
        try:
            merge_count = len(get_base_merge_files(".","dispatch_*/merge_*"))
        except:
            merge_count = -1

        timeout_wait = 60.0 #seconds
        while still_waiting and timeout_wait > 0.0: #could sleep until the entire job times out if there is a problem
            merge_list = get_base_merge_files(".","dispatch_*/*intermediate_merge.working")
            if len(merge_list) > 0:
                print(f"Waiting on *working files to complete ...")
                time.sleep(10.0) #sleep 10 secs
                timeout_wait = 60.0 #seconds #reset to full timeout
            else:
                #get the final list
                merge_list = get_base_merge_files(".", "dispatch_*/*intermediate_merge.h5")
                if len(merge_list) < merge_count: #the final job could get kicked off before all the prior jobs are complete or even started
                    print(f"Waiting on merge_count to match. Current {len(merge_list)}, expected {merge_count} ...")
                    time.sleep(10.0)             #so all the .working files might not even have been created yet
                    timeout_wait -= 10.0
                else:
                    still_waiting = False
                    print(f"Final Merge List {len(merge_list)}: {merge_list}")

        if still_waiting: #we timed out
            print(f"Timeout {timeout_wait:0.1f}s waiting on expected number {len(merge_list)}/{merge_count} of *intermediate_merge.h5 files. Aborting run.")
            dummy_list = [f"dispatch_{str(n).zfill(4)}/elixer_intermediate_merge.h5" for n in range(merge_count)]
            missing = np.setdiff1d(dummy_list,merge_list)
            print(f"Missing files: {missing}")
            with open("elixer_merged_cat.fail","w") as f:
                f.write(f"Timeout {timeout_wait:0.1f}s waiting on expected number {len(merge_list)}/{merge_count} of *intermediate_merge.h5 files. Aborting run.\n")
                f.write(f"Missing files: {missing}\n")
                f.write(f"Expected files: {dummy_list}\n")
                f.write(f"Found files: {merge_list}\n")
            exit(-1)

        if len(merge_list) > 0:
            needs_unique = False

            if os.path.exists("elixer_merged_cat.h5"):
                needs_unique = True
                merge_hdf5(merge_list, "elixer_merged_cat_new.h5")
                os.rename("elixer_merged_cat.h5","elixer_merged_cat_old.h5")
                merge_unique("elixer_merged_cat.h5","elixer_merged_cat_old.h5","elixer_merged_cat_new.h5")
                #merge_list.insert(0,"elixer_merged_cat.h5")
            else:
                merge_hdf5(merge_list, "elixer_merged_cat.h5")

            #clean up
            print("Cleaning up ...")
            for file in merge_list:
                if file != "elixer_merged_cat.h5":
                    print(f"Removing {file}")
                    os.remove(file)

            if os.path.exists("elixer_merged_cat_new.h5"):
                os.remove("elixer_merged_cat_new.h5")

            if os.path.exists("elixer_merged_cat_old.h5"):
                os.remove("elixer_merged_cat_old.h5")

    else: #this is a generational merge

        #this could be a re-run
        #first check if intermediate.working exists ... if so, delete it and continue
        #if it does not exist, see if intermetediate.h5 exists, if so, this one is already done so skip

        do_merge = True
        try:
            if os.path.exists("elixer_intermediate_merge.working"):
                os.remove("elixer_intermediate_merge.working")
                print(f"{merge_list_fn}: Incomplete. Delete working file and re-run.")
            elif not os.path.exists("elixer_intermediate_merge.h5"):
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

            print(f"Merging {len(merge_list)} files ... ")
            print(merge_list)
            merge_hdf5(merge_list)
            print("Intermediate merge complete")


if __name__ == '__main__':
    main()