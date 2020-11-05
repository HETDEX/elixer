"""
Plot the P(LyA)
and P(LAE)/P(OII) (aggregate)
by field and by catalog

a detection only has one field, but might have more than one catalog, so pick the deepest


"""
import numpy as np
import matplotlib.pyplot as plt
import tables

from astropy.table import Table


#first try to load from numpy saved files, if that fails, build new

reload = False


try:
    detectids = np.load("plot_detectids.npy")
    fieldnames = np.load("plot_fields.npy")
    p_lya = np.load("plot_p_lya.npy")
    plae_poii = np.load("plot_plae_poii.npy")
    agreement = np.load("plot_agree.npy")
    catalogs = np.load("plot_catalogs.npy",allow_pickle=True)
except Exception as e:
    print(e)
    print("did not find npy files, reloading ...")
    reload = True

if reload:
    t = Table.read("/work/05350/ecooper/wrangler/hdr2.1-detects/detect_hdr2.1.1.tab", format="ascii")
    curated_detects = t['detectid']

    h5 = tables.open_file("/media/dustin/Seagate8TB/hetdex/hdr2.1/elixer.h5")
    dtb = h5.root.Detections
    atb = h5.root.Aperture

    #skip the -1 classifications (spurious)
    detectids = [] #dtb.read_where("plae_classification >= 0",field="detectid")
    fieldnames = [] #dtb.read_where("plae_classification >= 0",field="fieldname")
    p_lya = [] #dtb.read_where("plae_classification >= 0",field="plae_classification")
    plae_poii = []#dtb.read_where("plae_classification >= 0",field="combined_plae")
    catalogs = []
    agreement = []
    bad_detectids = []

    nan_p_lya = []
    nan_plae_poii = []

    total_size = len(curated_detects)

    for i,det_id in enumerate(curated_detects):

        # sanity check
        if not (len(detectids) == len(catalogs) == len(fieldnames) == len(p_lya) == len(plae_poii) == len(
                agreement)):
            print(f"WFT?? array lengths mismatch {len(detectids)} vs {len(catalogs)} vs {len(fieldnames)}"
                  f"vs {len(p_lya)} vs {len(plae_poii)} vs {len(agreement)}")
            print(f"At detectid {det_id}")
            exit()

        if i % 10000 == 0:
            print(i, total_size)

        rows = dtb.read_where("detectid==det_id")
        if rows is None or len(rows) != 1:
            print(f"Bad read?? detectid {det_id}")
            continue

        if rows[0]['plae_classification'] < 0:
            print(f"Bad classsification detectid {det_id}")
            bad_detectids.append(det_id)
            continue

        if np.isnan(rows[0]['plae_classification']):
            print(f"NaN P(LyA) {det_id}")
            nan_p_lya.append(det_id)
            continue

        if np.isnan(rows[0]['combined_plae']):
            print(f"NaN P(LAE)/P(OII) {det_id}")
            nan_plae_poii.append(det_id)
            continue



        atb_rows = atb.read_where("detectid==det_id")
        if (atb_rows is None) or (len(atb_rows)==0):
            #catalogs.append(b"")
            print(f"No aperture, results unreliable, skipping {det_id}")
        else:

            detectids.append(det_id)
            fieldnames.append(rows[0]['fieldname'])
            p_lya.append(rows[0]['plae_classification'])
            plae_poii.append(rows[0]['combined_plae'])

            cu = np.unique(atb_rows['catalog_name'])
            if (cu is None) or len(cu) ==0:
                #this should not happend
                print("WFT?? no unique catalog names")
                catalogs.append(b"")
            elif len(cu) == 1:
                catalogs.append(cu[0])
            else:
                if b'CANDELS/EGS/CFHT' in cu:
                    catalogs.append(b'CANDELS/EGS/CFHT')
                elif b'GOODS-N' in cu:
                    catalogs.append(b'GOODS-N' )
                elif b'HyperSuprimeCam' in cu:
                    catalogs.append(b'HyperSuprimeCam' )
                elif b'KPNO' in cu:
                    catalogs.append(b'KPNO' )
                elif b'STACK_COSMOS' in cu:
                    catalogs.append(b'STACK_COSMOS' )
                elif b'DECAM/SHELA' in cu:
                    catalogs.append(b'DECAM/SHELA' )
                elif b'DECaLS' in cu:
                    catalogs.append(b'DECaLS' )
                elif b'Pan-STARRS' in cu:
                    catalogs.append(b'Pan-STARRS' )
                else:
                    #this should not happen
                    print("WTF?? No matching catalog name??")
                    catalogs.append(b"")

            #do PLAE/POII and P(LyA) agree?
            if plae_poii[-1] < 0.75: #OII suggested
                if p_lya[-1] < 0.25:
                    agreement.append(1)
                elif p_lya[-1] < 0.75:
                    agreement.append(0)
                else:
                    agreement.append(-1)
            elif plae_poii[-1] < 3.0: #neither suggested
                if p_lya[-1] < 0.25:
                    agreement.append(0)
                elif p_lya[-1] < 0.75:
                    agreement.append(1)
                else:
                    agreement.append(0)
            else: #LyA suggested
                if p_lya[-1] < 0.25:
                    agreement.append(-1)
                elif p_lya[-1] < 0.75:
                    agreement.append(0)
                else:
                    agreement.append(1)

    detectids = np.array(detectids)
    fieldnames = np.array(fieldnames)
    p_lya = np.array(p_lya)
    plae_poii = np.array(plae_poii)
    catalogs = np.array(catalogs)
    agreement = np.array(agreement)

    nan_p_lya = np.array(nan_p_lya)
    nan_plae_poii = np.array(nan_plae_poii)


    #write out
    np.save("plot_detectids",detectids)
    np.save("plot_fields",fieldnames)
    np.save("plot_catalogs",catalogs)
    np.save("plot_p_lya",p_lya)
    np.save("plot_plae_poii",plae_poii)
    np.save("plot_agree",agreement)

    # Nan
    if len(nan_plae_poii) > 0:
        np.savetxt("nan_plae_poii.list", nan_plae_poii, fmt="%d")

    if len(nan_p_lya) > 0:
        np.savetxt("nan_plae_poii.list", nan_p_lya, fmt="%d")

#sanity check
if not (len(detectids) == len(catalogs) == len(fieldnames) == len(p_lya) == len(plae_poii) == len(agreement)):
    print(f"WFT?? array lengths mismatch {len(detectids)} vs {len(catalogs)} vs {len(fieldnames)}"
          f"vs {len(p_lya)} vs {len(plae_poii)} vs {len(agreement)}")
    exit()

#bad_dets
# if len(bad_detectids) > 0:
#     with open("bad_dets.list","w") as f:
#         for d in bad_detectids:
#             f.write(f"{d}\n")
#     exit()




#search for disagrements:
sel1 = np.where(agreement == -1)[0]

#todo: split into two list: high PLAE vs low PlyA and the other way round
if sel1 is not None and len(sel1)>0:
    sel_bottom = np.where(p_lya < 0.25)[0]
    sel_top = np.where(p_lya > 0.75)[0]
    sel_left = np.where(plae_poii < 0.01)[0]
    sel_right = np.where(plae_poii > 100.0)[0]

    agreement_top_left = np.intersect1d(sel_top, sel_left)
    agreement_bottom_right = np.intersect1d(sel_bottom,sel_right)

    #agreement_top_left = np.intersect1d(np.intersect1d(sel1,sel_top), sel_left)
    #agreement_bottom_right = np.intersect1d(np.intersect1d(sel1,sel_bottom),sel_right)

    np.savetxt("agreement_0.list",detectids[sel1],fmt="%d")
    np.savetxt("agreement_top_left.list", detectids[agreement_top_left], fmt="%d")
    np.savetxt("agreement_bottom_right.list", detectids[agreement_bottom_right], fmt="%d")



plot_p_lya_bins = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
plot_plae_poii_bins = [0,0.25,0.5,0.75,1.,3.,5.,10.,50.,100.,1000.]
plot_log_plae_poii_bins = [-3,-2,-1,0,1,2,3]



unique_fields = np.unique(fieldnames)
catalogs[np.where(catalogs==None)] = b""
unique_catalogs = np.unique(catalogs)


#by field
plt.figure(figsize=(12,6))
plt.title("Fraction P(LyA) by Field")
plt.xlabel("P(LyA)")
plt.ylabel("Fraction")
for f in unique_fields:
    sel = np.where(fieldnames==f)[0]
    values = p_lya[sel]
    plt.hist(values,bins=plot_p_lya_bins,histtype='step', weights=np.ones(len(sel)) / len(sel),
             density=False,align='mid',label=f"{f.decode()} #{len(sel)}")


plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("hist_p_lya_by_field.png")



#by catalog
plt.close('all')
plt.figure(figsize=(12,6))
plt.title("P(LyA) by Catalog")
plt.xlabel("P(LyA)")
plt.ylabel("Fraction")
for f in unique_catalogs:
    sel = np.where(catalogs == f)[0]
    values = p_lya[sel]
    plt.hist(values, bins=plot_p_lya_bins, histtype='step',weights=np.ones(len(sel)) / len(sel),
             density=False, align='mid', label=f"{f.decode()} #{len(sel)}")

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("hist_p_lya_by_catalog.png")


#p_lay vs plae_poii
# agreement
# AND should just scatter plot with one on one axis


#scatter
#by field
plt.close('all')
plt.figure(figsize=(12,6))
plt.title("Log (PLAE/POII) vs P(LyA): All fields")
for f in unique_fields:
    sel = np.where(fieldnames == f)[0]
    values = p_lya[sel]
    plt.scatter(np.log10(plae_poii[sel]),p_lya[sel],s=0.01,label=f"{f.decode()}")

plt.xlabel("Log (PLAE/POII)")
plt.ylabel("P(LyA)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',markerscale=100)
plt.tight_layout()
plt.savefig("scatter_p_lya_vs_plae_poii_by_field.png")

#by catalog
plt.close('all')
plt.figure(figsize=(12,6))
plt.title("Log (PLAE/POII) vs P(LyA): All Catalogs")
for f in unique_catalogs:
    sel = np.where(catalogs == f)[0]
    values = p_lya[sel]
    plt.scatter(np.log10(plae_poii[sel]), p_lya[sel], s=0.01, label=f"{f.decode()}")

plt.xlabel("Log (PLAE/POII)")
plt.ylabel("P(LyA)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',markerscale=100)
plt.tight_layout()
plt.savefig("scatter_p_lya_vs_plae_poii_by_catalog.png")



#scatters, individually
for f in unique_fields:
    plt.close('all')
    plt.figure(figsize=(12, 6))
    plt.title(f"Log (PLAE/POII) vs P(LyA): Field {f.decode()} ")

    sel = np.where(fieldnames == f)[0]
    values = p_lya[sel]
    size = 0.1
    if len(sel) < 10000:
        size = 0.5
    plt.scatter(np.log10(plae_poii[sel]),p_lya[sel],s=size)

    plt.xlabel("Log (PLAE/POII)")
    plt.ylabel("P(LyA)")
    plt.tight_layout()
    plt.savefig("scatter_p_lya_vs_plae_poii_field_" + f.decode().replace("/","-")  +".png")

for f in unique_catalogs:
    plt.close('all')
    plt.figure(figsize=(12, 6))
    plt.title(f"Log (PLAE/POII) vs P(LyA): Catalog {f.decode()} ")

    sel = np.where(catalogs == f)[0]
    size = 0.1
    if len(sel) < 10000:
        size = 0.5
    values = p_lya[sel]
    plt.scatter(np.log10(plae_poii[sel]), p_lya[sel], s=size)

    plt.xlabel("Log (PLAE/POII)")
    plt.ylabel("P(LyA)")
    plt.tight_layout()

    plt.savefig("scatter_p_lya_vs_plae_poii_catalog_" + f.decode().replace("/","-") + ".png")

#histogram

plt.close('all')
plt.figure()
plt.title("P(LyA) vs  P(LAE)/P(OII) Agreement")
values = agreement
plt.hist(values,bins=3)
plt.tight_layout()
plt.savefig("hist_agreement.png")





#
# P(LAE)/P(OII)
#

#by field
plt.close('all')
plt.figure(figsize=(12,6))
plt.title("Fraction P(LAE)/P(OII) by Field")
plt.xlabel("Log P(LAE)/P(OII)")
plt.ylabel("Fraction")
for f in unique_fields:
    sel = np.where(fieldnames==f)[0]
    values = plae_poii[sel]
    plt.hist(np.log10(values),bins = plot_log_plae_poii_bins, histtype='step',weights=np.ones(len(sel)) / len(sel),
             density=False,align='mid',label=f"{f.decode()} #{len(sel)}")


plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("hist_plae_poii_by_field.png")



#by catalog
plt.close('all')
plt.figure(figsize=(12,6))
plt.title("P(LAE)/P(OII) by Catalog")
plt.xlabel("Log P(LAE)/P(OII)")
plt.ylabel("Fraction")
for f in unique_catalogs:
    sel = np.where(catalogs == f)[0]
    values = plae_poii[sel]
    plt.hist(np.log10(values),histtype='step', bins = plot_log_plae_poii_bins, weights=np.ones(len(sel)) / len(sel),
             density=False, align='mid', label=f"{f.decode()} #{len(sel)}")

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("hist_plae_poii_by_catalog.png")