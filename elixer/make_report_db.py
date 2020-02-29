import sys
sys.path.append("/home/dustin/code/python/hetdex_api/hetdex_api")
sys.path.append("/work/03261/polonius/wrangler/code/hetdex_api/hetdex_api")
import sqlite_utils as sql
import argparse


def parse_commandline(auto_force=False):
    desc = "make a single ELiXer report database (SQLite3)"



    parser = argparse.ArgumentParser(description=desc)


    parser.add_argument('--db_name', help="Database name (output)", required=True, type=str)
    parser.add_argument('--img_dir', help="Directory with the images", required=True, type=str)
    parser.add_argument('--img_name', help="Wildcard image name", required=True, type=str)

    args = parser.parse_args()


    return args


def main():
    #go blind?
    args = parse_commandline()

    sql.build_elixer_report_image_db(args.db_name,args.img_dir,args.img_name)



if __name__ == '__main__':
    main()