#the noglob is IMPORTANT and it will stick after this is run IF you source
# (if you bash instead it will not stick)
# if running in SLURM, the environment from which you launched must have had
# the noglob applied otherwise the SLURM launcher will expand the '*' in the call and
# not pass them as parameters
set -o noglob

imgdir="/data/03261/polonius/hdr2.1.run/elixer/rerun/rerun1/"
db_dir="/data/03261/polonius/hdr2.1.run/detect/image_db/"
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
#for i in {21000..21005}
#for i in {21006..21010}
#for i in {21011..21015}
#for i in {21016..21020}
#for i in {21021..21025}
#for i in {21900..21900}
#do
#
#  python3 make_report_db.py --db_name "${db_dir}${rpt}${i}.db" --img_dir $imgdir --img_name "${i}*[0-9].png" &
#
#  #note: no underscore in image names -- of form 2000123456nei.png
#  #      with elixer v 1.8.x and up, yes, underscore with nei (2100123456_nei.png)
#  #note: YES underscore in db names
#  python3 make_report_db.py --db_name "${db_dir}${rpt}${i}_nei.db" --img_dir  $imgdir  --img_name "${i}*_nei.png" &
#
#  #note: YES underscore in image names -- of form 2000123456_mini.png
#  #note: YES underscore in db names
#  python3 make_report_db.py --db_name "${db_dir}${rpt}${i}_mini.db" --img_dir  $imgdir  --img_name "$i*[0-9]_mini.png" &
#
#done


#################################
# -OR-
#
# use this section below to create an
# elixer_reports.run file for the pre-made
# elixer_reports.slurm file
####################################

for i in {21000..21025}
do

  echo python3 make_report_db.py --db_name "${db_dir}${rpt}${i}.db" --img_dir $imgdir --img_name "${i}*[0-9].png"

  #note: no underscore in image names -- of form 2000123456nei.png
  #      with elixer v 1.8.x and up, yes, underscore with nei (2100123456_nei.png)
  #note: YES underscore in db names
  echo python3 make_report_db.py --db_name "${db_dir}${rpt}${i}_nei.db" --img_dir  $imgdir  --img_name "${i}*_nei.png"

  #note: YES underscore in image names -- of form 2000123456_mini.png
  #note: YES underscore in db names
  echo python3 make_report_db.py --db_name "${db_dir}${rpt}${i}_mini.db" --img_dir  $imgdir  --img_name "$i*[0-9]_mini.png"

done
