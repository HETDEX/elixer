# the noglob is IMPORTANT and it will stick after this is run IF you source
# (if you bash instead it will not stick)
# if running in SLURM, the environment from which you launched must have had
# the noglob applied otherwise the SLURM launcher will expand the '*' in the call and
# not pass them as parameters
set -o noglob

imgdir="/data/03261/polonius/hdr3/d##/all_pngs"
#!!! IMPORTANT !!! db_dir needs to end with a /
db_dir="/data/03261/polonius/hdr3/detect/image_db/"
rpt="elixer_reports_"


#############################################
# uncommment the next section to run
# in an idev session:
# idev -A Hobby-Eberly-Telesco -t 20:00:00
############################################

#echo Running all prefixes serially in the foreground.
#echo This updates databases in place. No copy.
#
#read -p "Press enter to continue"
#
#at 48 cores (SKX), and 3 db (rpt, nei, mini) per 100,000 detectID numbering, run 16 per node
#at 68 cores (KNL), and 3 db (rpt, nei, mini) per 100,000 detectID numbering, run 22 per node
#takes about 20 hours (or it did) for 100,000 ACTUAL inserts per file ...
#this should not take as long as there are usually only order few to 10,000 actual detections n a 100,000 numbering range
#  the rest do not meet the threshold for a report
for i in {30000..30015}
#for i in {30900..30900}
do

  python3 make_report_db.py --db_name "${db_dir}${rpt}${i}.db" --img_dir $imgdir --img_name "${i}*[0-9].png" &

  #note: no underscore in image names -- of form 2000123456nei.png
  #      with elixer v 1.8.x and up, yes, underscore with nei (2100123456_nei.png)
  #note: YES underscore in db names
  python3 make_report_db.py --db_name "${db_dir}${rpt}${i}_nei.db" --img_dir  $imgdir  --img_name "${i}*_nei.png" &

  #note: YES underscore in image names -- of form 2000123456_mini.png
  #note: YES underscore in db names
  python3 make_report_db.py --db_name "${db_dir}${rpt}${i}_mini.db" --img_dir  $imgdir  --img_name "$i*_mini.png" &

done


#################################
# -OR-
#
# use this section below to create an
# elixer_reports.run file for the pre-made
# elixer_reports.slurm file
####################################

for i in {30000..30015}
#for i in {30900..30900}
do

  echo python3 make_report_db.py --db_name "${db_dir}${rpt}${i}.db" --img_dir $imgdir --img_name "${i}*[0-9].png"

  #note: no underscore in image names -- of form 2000123456nei.png
  #      with elixer v 1.8.x and up, yes, underscore with nei (2100123456_nei.png)
  #note: YES underscore in db names
  echo python3 make_report_db.py --db_name "${db_dir}${rpt}${i}_nei.db" --img_dir  $imgdir  --img_name "${i}*_nei.png"

  #note: YES underscore in image names -- of form 2000123456_mini.png
  #note: YES underscore in db names
  echo python3 make_report_db.py --db_name "${db_dir}${rpt}${i}_mini.db" --img_dir  $imgdir  --img_name "$i*_mini.png"

done
