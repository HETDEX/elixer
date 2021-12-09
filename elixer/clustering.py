"""
Scans elixer h5 file for neighbors with matching emission lines to cluster at a common redshift
The intent is largely to resolve redshift mis-classifications as high-z that are actually single emission lines
in the outskirts of a bright object where the aperture fails to get significant continuum

"""


from __future__ import print_function
import logging
import numpy as np
import os.path as op

try:
    from elixer import global_config as G
    from elixer import utilities
    from elixer import spectrum
except:
    import global_config as G
    import utilities
    import spectrum

log = G.Global_Logger('clustering')
log.setlevel(G.LOG_LEVEL)



def find_cluster(detectid,elixerh5,outfile=True,delta_arcsec=15.0,delta_lambda=2.0,gmag_thresh=23.0):
    """

    Note: caller is responisble for any pre-checks on the target detectid and for opening and closing the h5 file

    :param detectid: detectid of the target HETDEX detection
    :param elixerh5: handle to the elixer h5 file to search
    :param outfile: if TRUE writes out a file with the clutering info (if a cluster is made) named as detectid.cluster
    :param delta_arcsec: +/- width in arsecs to search in RA and in Dec
    :param delta_lambda: +/- width to search in found emission lines
    :param gmag_thresh:  faint limit gmag to allow as a match (we only want to match against bright objects)
    :return: cluster dictionary with key info
    """

    cluster_dict = None
    # {"detectid": detectid,
    #      "neighborid": None, #detectID of the neighbor
    #      "neighbor_dist": None, #distance to HETDEX ra, dec of neighbor
    #      "neighbor_ra": None,
    #      "neighbor_dec": None,
    #      "neighhor_gmag": None,
    #      "neighbor_z": None,
    #      "neighbor_qz": None,
    #      "neighbor_plya": None
    #      }

    try:
        log.info(f"Clustering on {detectid} ...")

        dtb = elixerh5.root.Detections
        ltb = elixerh5.root.SpectraLines

        #first get the detectid related info
        q_detectid = detectid
        rows = dtb.read_where("detectid == q_detectid")
        if len(rows) != 1:
            log.info(f"Invalid detectid {detectid}")
            return cluster_dict

        #flags = rows[0][''] #could explicitly check for a magnitude mismatch

        try:
            if rows[0]['review'] == 0: #if we are NOT set to review, check the gmag
                target_gmag = rows[0]['mag_sdss_g'] #this could fail
                target_gmag_err = rows[0]['mag_sdss_g_err'] #this could fail
                try:
                    if (target_gmag+target_gmag_err) < gmag_thresh: #too bright
                        log.info(f"Invalid detectid {detectid}. Too bright. gmag = {target_gmag} +/- {target_gmag_err}")
                        return cluster_dict
                except: #the sdss might not be there or may be invalid
                    target_gmag = rows[0]['mag_full_spec'] #this could fail
                    target_gmag_err = rows[0]['mag_full_spec_err'] #this could fail
                    if (target_gmag+target_gmag_err) < gmag_thresh: #too bright
                        log.info(f"Invalid detectid {detectid}. Too bright. gmag = {target_gmag} +/- {target_gmag_err}")
                        return cluster_dict
        except:
            pass #older ones may not have a 'review' field

        target_ra = rows[0]['ra']
        target_dec = rows[0]['dec']
        target_z = rows[0]['best_z']
        target_wave = rows[0]['wavelength_obs']
        target_wave_err = rows[0]['wavelength_obs_err']

        deg_err = delta_arcsec / 3600.0

        #box defined by COORDINATEs not by real delta_arcsec .... ie.. no Dec correction to RA
        ra1 = target_ra - deg_err
        ra2 = target_ra + deg_err
        dec1 = target_dec - deg_err
        dec2 = target_dec + deg_err


        #now search for RA, Dec neighbors
        #there is an index on ra and dec
        rows = dtb.read_where("(ra > ra1) & (ra < ra2) & (dec > dec1) & (dec < dec2)")

        if len(rows) == 0: #there are none
            log.info(f"Clustering on {detectid}. No neighbors found.")
            return cluster_dict

        #otherwise, check for other conditions
        #gmag limit
        sel = np.array(rows['mag_sdss_g'] < gmag_thresh) | np.array(rows['mag_full_spec'] < gmag_thresh)
        if np.sum(sel) == 0:
            log.info(f"Clustering on {detectid}. No neighbors meet minimum requirements.")
            return cluster_dict

        rows = rows[sel]

        #check lines
        neighbor_ids = rows['detectid']
        line_scores = np.zeros(len(neighbor_ids))
        line_w_obs = np.zeros(len(neighbor_ids))
        used_in_solution = np.full(len(neighbor_ids),False)

        w1 = target_wave - target_wave_err - delta_lambda
        w2 = target_wave + target_wave_err + delta_lambda
        sel = np.full(len(neighbor_ids),True)

        for i,id in enumerate(neighbor_ids):
            lrows = ltb.read_where("(detectid==id) & (sn > 4.5) & (score > 5.0) & (wavelength > w1) & (wavelength < w2)")
            if len(lrows) == 0:
                sel[i] = False
                continue
            line_scores[i] = np.max(lrows['score'])
            line_w_obs[i] = lrows[np.argmax(lrows['score'])]['wavelength']
            used_in_solution[i] = lrows[np.argmax(lrows['score'])]['used'] #NOTE: this might not be a multiline solution
                                                                           #in which case, used can be False

        if np.sum(sel) == 0:
            log.info(f"Clustering on {detectid}. No neighbors meet minimum emission line requirements.")
            return cluster_dict

        #now choose the "best" one from those that remain
        rows = rows[sel]

        #best could be brightest? or highest score on the matching line?
        brightest = np.argmin(rows['mag_sdss_g'])
        best_line = np.argmax(line_scores)
        #best_pz = np.argmax(rows['best_pz'])

        #take brightest unless the best_line does not match and is more than 25% better?
        best_idx = brightest
        if brightest != best_line:
            if  line_scores[brightest] / best_line < 0.75:
                best_idx = best_line


        #check if the z is the same, then don't bother
        if abs(rows[best_idx]['best_z'] - target_z) < 0.1:
            log.info(f"Clustering on {detectid}. Neighbors at same z = {target_z:0.5f}")
            return cluster_dict

        #check that the emission line IS USED in the solution
        #or if not used, that it is CONSISTENT with the solution
        if not used_in_solution[best_idx]:
            sp = spectrum.Spectrum()
            lines = sp.match_lines(line_w_obs[best_idx],
                                   rows[best_idx]['best_z'],
                                   z_error=0.001,
                                   aa_error=None,
                                   allow_absorption=False)
            if lines is None or len(lines) == 0:
                log.info(f"Clustering on {detectid}. Best neighbor {neighbor_ids[best_idx]} line {line_w_obs[best_idx]:0.2f} inconsistent with redshift {rows[best_idx]['best_z']:0.4f}."
                         f"No common lines near rest {line_w_obs[best_idx]/(1 + rows[best_idx]['best_z']):0.2f}")
                return cluster_dict



        #now populate the dictionary
        try:
            plya = rows[best_idx]['plya_classification']
        except:
            plya = rows[best_idx]['plae_classification']

        cluster_dict = {"detectid": detectid,
                            "neighborid": rows[best_idx]['detectid'], #detectID of the neighbor
                            "neighbor_dist": utilities.angular_distance(target_ra,target_dec,rows[best_idx]['ra'],rows[best_idx]['dec']), #distance to HETDEX ra, dec of neighbor
                            "neighbor_ra": rows[best_idx]['ra'],
                            "neighbor_dec": rows[best_idx]['dec'],
                            "neighhor_gmag": rows[best_idx]['mag_sdss_g'],
                            "neighbor_z": rows[best_idx]['best_z'],
                            "neighbor_qz": rows[best_idx]['best_pz'],
                            "neighbor_plya": plya
                            }

        log.info(f"Clustering on {detectid}. Found bright neighbor ({rows[best_idx]['detectid']}) at z = {rows[best_idx]['best_z']:0.5f}")
        if outfile:
            with open(f"{detectid}.cluster","w+") as f:
                f.write("# detectid  n_z      n_qz  n_detectid  n_ra       n_dec     n_dist  n_gmag  n_p(lya)\n")
                f.write(f"{detectid}  {cluster_dict['neighbor_z']:0.5f}  {cluster_dict['neighbor_qz']:0.2f}  {cluster_dict['neighborid']}  "
                        f"{cluster_dict['neighbor_ra']:0.5f}  {cluster_dict['neighbor_dec']:0.5f}  {cluster_dict['neighbor_dist']:0.2f}    "
                        f"{cluster_dict['neighhor_gmag']:0.2f}   {cluster_dict['neighbor_plya']:0.2f}\n")



    except:
        log.error("Exception! Excpetion in clustering::find_cluster()",exc_info=True)

    return cluster_dict


def cluster_multiple_detectids(detectid_list,elixerh5,outfile=True,delta_arcsec=15.0,delta_lambda=2.0,gmag_thresh=23.0):
    """
    Wraper for find_cluster that takes a list of detectids instead

    :param detectid_list:
    :param elixerh5:
    :param outfile:
    :param delta_arcsec:
    :param delta_lambda:
    :param gmag_thresh:
    :return:
    """

    cluster_list = []

    try:
        for d in detectid_list:
            try:
                cluster_dict = find_cluster(d,elixerh5,outfile,delta_arcsec,delta_lambda,gmag_thresh)
                if cluster_dict is not None:
                    cluster_list.append(cluster_dict)
            except:
                log.error("Exception! Exception iterating in clustering::cluster_multiple_detectids().",exc_info=True)
    except:
        log.error("Exception! Exception in clustering::cluster_multiple_detectids().",exc_info=True)

    return cluster_list


def cluster_all_detectids(elixerh5,outfile=True,delta_arcsec=15.0,delta_lambda=2.0,gmag_thresh=23.0):
    """
    Wraper for find_cluster that takes a list of detectids instead

    :param detectid_list:
    :param elixerh5:
    :param outfile:
    :param delta_arcsec:
    :param delta_lambda:
    :param gmag_thresh:
    :return:
    """

    cluster_list = []

    try:

        detectid_list = elixerh5.root.Detections.read(field="detectid")

        for d in detectid_list:
            try:
                cluster_dict = find_cluster(d,elixerh5,outfile,delta_arcsec,delta_lambda,gmag_thresh)
                if cluster_dict is not None:
                    cluster_list.append(cluster_dict)
            except:
                log.error("Exception! Exception iterating in clustering::cluster_multiple_detectids().",exc_info=True)
    except:
        log.error("Exception! Exception in clustering::cluster_multiple_detectids().",exc_info=True)

    return cluster_list