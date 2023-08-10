# give a directory with *.dets files, create an elixer run directory for each, complete with .slurm files
# BUT do not queue the slurm files
# User must later run: sbatch for each of the slurms, with dependencies as needed

dets=$(ls [cd]??_*[0-9].dets)
nodes=8
time="01:59:59"

for det in $dets
do
  selixer  -f --slurm 0 --log error --hdr 4 --name ${det:0:3} --dets ${det} --png --mini --plya_thresh 0.5,0.4,0.3 --check_z 255 --neighborhood 10 --nodes ${nodes} --time ${time} --tasks 0 --require_hetdex --email dustin@astro.as.utexas.edu
done

