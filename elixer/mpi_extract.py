from mpi4py import MPI
import astropy.units as u
from astropy.coordinates import SkyCoord
from hetdex_tools.get_spec import get_spectra as hda_get_spectra
from hetdex_api.shot import get_fibers_table as hda_get_fibers_table
from astropy.table import QTable, Table, Column


#edit here get info we want to extract
ra,dec,shot = np.loadtxt("random_coords.txt",unpack=True,skiprows=1)


class DEX_Coord():
    def __inti__():
        pass

    ra = None
    dec = None
    shot = None

class spec_bundle:
    def __init__():
        pass

    spec_ff = []
    spec_lo = []
    err_ff = []
    err_lo = []
    weights_ff = []
    weights_lo = []
    flag = []
    gal_flag = []
    meteor_flag = []
    amp_flag = []



def pool_specs(dex_coords):

    spec_ff = []
    spec_lo = []
    err_ff = []
    err_lo = []
    weights_ff = []
    weights_lo = []
    flag = []
    gal_flag = []
    meteor_flag = []
    amp_flag = []

    result_list = []

    for dc in dex_coords:
        ra = dc.ra
        dec = dc.dec
        shot = dc.shot

        try:
            coord = SkyCoord(ra=r * u.deg, dec=d * u.deg)
            apt_ff = hda_get_spectra(coord, survey=f"hdr2.1", shotid=s,
                                     ffsky=True, multiprocess=False, rad=3.0,
                                     tpmin=0.0,fiberweights=False) #don't need the fiber weights
            apt_lo = hda_get_spectra(coord, survey=f"hdr2.1", shotid=s,
                                     ffsky=False, multiprocess=False, rad=3.0,
                                     tpmin=0.0,fiberweights=False) #don't need the fiber weights

            if len(apt_ff)==1:
                spec_ff = copy.copy(apt_ff['spec'][0]*2.0)
                err_ff = copy.copy(apt_ff['spec_err'][0]*2.0)
                weights_ff = copy.copy(apt_ff['weights'][0])
                flag = apt_ff['flag'][0]
                gal_flag = apt_ff['gal_flag'][0]
                amp_flag = apt_ff['amp_flag'][0]
                meteor_flag = apt_ff['meteor_flag'][0]

            else:
                spec_ff = np.zeros(1036)
                err_ff = np.zeros(1036)
                weights_ff = np.zeros(1036)

            if len(apt_lo)==1:
                spec_lo = copy.copy(apt_lo['spec'][0]*2.0)
                err_lo = copy.copy(apt_lo['spec_err'][0]*2.0)
                weights_lo = copy.copy(apt_ff['weights'][0])
            else:
                spec_lo = np.zeros(1036)
                err_lo = np.zeros(1036)
                weights_lo = np.zeros(1036)

            sb = spec_bundle()
            sb.spec_ff = spec_ff
            sb.spec_lo = spec_lo

            sb.err_ff = err_ff
            sb.err_lo = err_lo
            sb.weights_ff = weights_ff
            sb.weights_lo = weights_lo
            sb.flag = flag
            sb.gal_flag = gal_flag
            sb.meteor_flag = meteor_flag
            sb.amp_flag = amp_flag

            result_list.append(sb)
        except Exception as e:
            pass






    #     result_dict = {}
    #     result_dict['spec_ff'] = spec_ff
    #     result_dict['spec_lo'] = spec_lo
    #     result_dict['err_ff'] = err_ff
    #     result_dict['err_lo'] = err_lo
    #     result_dict['weights_ff'] = weights_ff
    #     result_dict['weights_lo'] = weights_lo
    #     result_dict['flag'] = flag
    #     result_dict['gal_flag'] = gal_flag
    #     result_dict['meteor_flag'] = meteor_flag
    #     result_dict['amp_flag'] = amp_flag
    #    print("done")
    return result_list



def main():

    COMM = MPI.COMM_WORLD
    SIZE = COMM.Get_size()
    RANK = COMM.Get_rank()
    NAME = MPI.Get_processor_name()

    COORD_CHUNKS = []
    #just for learning purposes
    #in this specific case, could skip this entirely and just have each worker use its RANK as the seed
    #as well as the total sims to run ... since this is previously agreed on (as a global constant)
    seed = None
    if (RANK == 0):
        # MPI start the clock
        walltime = MPI.Wtime()
        # print ("Start time:", walltime)

        #build the coordinates to use, should already be read in
        dex_coords = []
        for r,d,s in zip(ra,dec,shot):
            dc = DEX_Coord()
            dc.ra = r
            dc.dec = d
            dc.shot = s
            dex_coords.append(dc)

        #todo: divide up into chunks gor the number of workers
        COORD_CHUNKS = np.array_split(dex_coords, SIZE)

        #since the program has this defined, no need to actually communicate here

    #send everybody the total number to run
    #COMM.bcast(TOTAL_SIMS_TO_RUN)

    #send each core its own chunk
    COMM.scatter(COORD_CHUNKS)


    #ALL cores send results to manager
    recvbuf = None
    if RANK == 0:
        recvbuf = np.empty([SIZE, SIZE_OF_WAVEBINS], dtype=int)

    #I think there is another way to do this ... like a Reduce call?

    #gather up everyone's n_esc
    COMM.Gather(n_esc, recvbuf, root=0)

    #sum up ... manager
    sum = []
    count = 0
    if RANK==0:
        sum = np.zeros(SIZE_OF_WAVEBINS)
        for i in range(SIZE):
            if DEBUG_PRINT:
                print("Summing", i, recvbuf[i])
            sum += recvbuf[i]

        if DEBUG_PRINT:
            print("SUM", sum)

        sum = np.array(sum).astype(float)

        f_esc = sum / (PHOTONS_PER_BIN * TOTAL_SIMS_TO_RUN)

        if DEBUG_PRINT:
            print("f_esc", f_esc)



    if RANK==0:
        # MPI stop the clock, get ellapsed time
        walltime = MPI.Wtime() - walltime

        print("Delta-time: ", walltime)
        print("   Per-sim: ", walltime/TOTAL_SIMS_TO_RUN)


    #todo: here is where we would make the plots, but don't bother since this is just
    #todo: a timing exercise

    if (RANK == 0):
        #sanity check .. make sure the escape curve matches the original data
        wavelengths = np.logspace(-2, 0.5, SIZE_OF_WAVEBINS)  # 10AA ~ 30,000AA
        plt.close('all')
        plt.plot(wavelengths, f_esc, label="mean f_esc")
        #plt.fill_between(wavelengths, f_esc_low, f_esc_high, color='k', alpha=0.3, label=r"1-$\sigma$")
        plt.xscale('log')
        plt.legend()
        plt.title("f_esc by wavelength (%d simulations)" % (TOTAL_SIMS_TO_RUN))
        plt.xlabel("wavelength bin [microns]")
        plt.ylabel("fraction of escaped photons")

        plt.savefig("rel_f_esc.png")

    MPI.Finalize()

if __name__ == '__main__':
    main()
