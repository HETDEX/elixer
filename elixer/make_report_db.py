#import sys
#sys.path.append("/home/dustin/code/python/hetdex_api/hetdex_api")
#sys.path.append("/work/03261/polonius/wrangler/code/hetdex_api/hetdex_api")
from hetdex_api import sqlite_utils as sql
import argparse
import shutil
import os


def parse_commandline(auto_force=False):
    desc = "make a single ELiXer report database (SQLite3)"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--db_name', help="Database name (output)", required=True, type=str)
    parser.add_argument('--img_dir', help="Directory with the images", required=True, type=str)
    parser.add_argument('--img_name', help="Wildcard image name", required=True, type=str)
    parser.add_argument('--mv2dir', help="mv to directory (leaves in cwd if not provided)",required=False,type=str)

    args = parser.parse_args()

    return args


def main():
    #go blind?
    args = parse_commandline()

    sql.build_elixer_report_image_db(args.db_name,args.img_dir,args.img_name)

    if args.mv2dir:
        if os.path.exists(args.mv2dir):
            if not os.path.exists(os.path.join(args.mv2dir,args.db_name)):
                shutil.move(args.db_name, args.mv2dir)
            else:
                print(f"{args.db_name} alread exists in mv2dir {args.mv2dir}. Will NOT overwrite.")
        else:
            print(f"{args.mv2dir} does not exist. Cannot move {args.db_name}.")

    print(f"{args.db_name} complete")

if __name__ == '__main__':
    main()