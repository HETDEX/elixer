set -o noglob
cp2dir="/data/03261/polonius/hdr2.1.run/detect/image_db/staging"
db_dir="./"
imgdir="all_pngs"
rpt="elixer_reports_"
nei="_nei"
mini="_mini"

#############################################
# uncommment the next section to run
# in an idev session:
# idev -A Hobby-Eberly-Telesco -t 20:00:00
############################################

#echo Running prefixes 0 to 12
#echo This copies databases to staging location.
#
#read -p "Press enter to continue"
#
#for i in {21000..21012}
#do
#  echo "$db_dir$rpt$i.db"
#
#  python3 make_report_db.py --db_name "$db_dir$rpt$i.db" --img_dir $imgdir --img_name "$i*[0-9].png" --mv2dir $cp2dir &
#
#  python3 make_report_db.py --db_name "$db_dir$rpt$i$nei.db" --img_dir  $imgdir  --img_name "$i*[0-9]_nei.png" --mv2dir $cp2dir &
#
#  python3 make_report_db.py --db_name "$db_dir$rpt$i$mini.db" --img_dir  $imgdir  --img_name "$i*[0-9]_mini.png" --mv2dir $cp2dir &
#
#done

#################################
# -OR-
#
# use this section below to create an
# elixer_reports.run file for the pre-made
# elixer_reports.slurm file
####################################

for i in {21000..21012}
do

  echo python3 make_report_db.py --db_name "$db_dir$rpt$i.db" --img_dir $imgdir --img_name "$i*[0-9].png" --mv2dir $cp2dir &

  echo python3 make_report_db.py --db_name "$db_dir$rpt$i$nei.db" --img_dir  $imgdir  --img_name "$i*[0-9]_nei.png" --mv2dir $cp2dir &

  echo python3 make_report_db.py --db_name "$db_dir$rpt$i$mini.db" --img_dir  $imgdir  --img_name "$i*[0-9]_mini.png" --mv2dir $cp2dir &

done
