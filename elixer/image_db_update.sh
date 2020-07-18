set -o noglob

imgdir="/data/03261/polonius/hdr2.run/rxx/dispatch*/*/"
db_dir="/data.03261/poloniu/hdr2.run/detect/image_db/"
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
#for i in {20000..20012}
#do
#
#  python3 make_report_db.py --db_name "${db_dir}${rpt}${i}.db" --img_dir $imgdir --img_name "${i}*[0-9].png"
#
#  #note: no underscore in image names -- of form 2000123456nei.png
#  #      with elixer v 1.8.x and up, yes, underscore with nei (2100123456_nei.png)
#  #note: YES underscore in db names
#  python3 make_report_db.py --db_name "${db_dir}${rpt}${i}_nei.db" --img_dir  $imgdir  --img_name "${i}*_nei.png"
#
#  #note: YES underscore in image names -- of form 2000123456_mini.png
#  #note: YES underscore in db names
#  python3 make_report_db.py --db_name "${db_dir}${rpt}${i}_mini.db" --img_dir  $imgdir  --img_name "$i*[0-9]_mini.png"
#
#done


#################################
# -OR-
#
# use this section below to create an
# elixer_reports.run file for the pre-made
# elixer_reports.slurm file
####################################

for i in {20000..20012}
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
