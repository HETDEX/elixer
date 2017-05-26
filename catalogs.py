import global_config as G

log = G.logging.getLogger('Cat_logger')
log.setLevel(G.logging.DEBUG)

#base class for catalogs (essentially an interface class)
#all Catalogs classes must implement:

import cat_candles_egs_stefanon_2016
import cat_egs_groth
import cat_stack_cosmos
import cat_shela


def get_catalog_list():
    #build list of all catalogs below
    cats = list()
    cats.append(cat_candles_egs_stefanon_2016.CANDELS_EGS_Stefanon_2016())
    #cats.append(EGS_GROTH()) #this is of no value right now
    cats.append(cat_stack_cosmos.STACK_COSMOS())
    cats.append(cat_shela.SHELA())

    return cats
