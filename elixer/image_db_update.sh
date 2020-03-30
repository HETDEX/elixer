set -o noglob

imgdir="/data/03261/polonius/hdr2.run/rxx/dispatch*/*/"
db_dir="/data.03261/poloniu/hdr2.run/detect/image_db/"
rpt="elixer_reports_"


echo Running all prefixes serially in the foreground.
echo This updates databases in place. No copy.

read -p "Press enter to continue"

for i in {20000..20012}
do

  python3 make_report_db.py --db_name "${db_dir}${rpt}${i}.db" --img_dir $imgdir --img_name "${i}*[0-9].png"

  #note: no underscore in image names -- of form 2000123456nei.png
  #note: YES underscore in db names
  python3 make_report_db.py --db_name "${db_dir}${rpt}${i}_nei.db" --img_dir  $imgdir  --img_name "${i}*nei.png"

  #note: YES underscore in image names -- of form 2000123456_mini.png
  #note: YES underscore in db names
  python3 make_report_db.py --db_name "${db_dir}${rpt}${i}_mini.db" --img_dir  $imgdir  --img_name "$i*[0-9]_mini.png"

done
