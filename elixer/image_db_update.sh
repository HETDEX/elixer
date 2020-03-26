imgdir="all_pngs"
db_dir="./"
rpt="elixer_reports_"
nei="_nei"
mini="_mini"

echo Running prefixes 0 to 12
echo This is an UPDATE \(append and replace\). No file copy performed.

read -p "Press enter to continue"

for i in {20000..20012}
do
  echo "$db_dir$rpt$i.db"
  
  python3 make_report_db.py --db_name "$db_dir$rpt$i.db" --img_dir $imgdir --img_name "$i*[0-9].png" &
  
  python3 make_report_db.py --db_name "$db_dir$rpt$i$nei.db" --img_dir  $imgdir  --img_name "$i*[0-9]nei.png" &
  
  python3 make_report_db.py --db_name "$db_dir$rpt$i$mini.db" --img_dir  $imgdir  --img_name "$i*[0-9]_mini.png" &

done
