import global_config as G

log = G.logging.getLogger('Cat_logger')
log.setLevel(G.logging.DEBUG)

#base class for catalogs (essentially an interface class)
#all Catalogs classes must implement:

import cat_candles_egs_stefanon_2016
import cat_goods_n
#import cat_goods_n_finkelstein
import cat_egs_groth
import cat_stack_cosmos
import cat_shela
import cat_catch_all


def get_catalog_list():
    #build list of all catalogs below
    cats = list()
    cats.append(cat_candles_egs_stefanon_2016.CANDELS_EGS_Stefanon_2016())
    cats.append(cat_goods_n.GOODS_N())
    #cats.append(cat_goods_n_finkelstein.GOODS_N_FINKELSTEIN())
    #cats.append(EGS_GROTH()) #this is of no value right now
    cats.append(cat_stack_cosmos.STACK_COSMOS())
    cats.append(cat_shela.SHELA())

    return cats

def get_catch_all():
    return cat_catch_all.CATCH_ALL()