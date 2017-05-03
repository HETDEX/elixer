#specific implementation of The CANDELS-EGS Multi-wavelength catalog Stefanon et al., 2016

#CandlesEgsStefanon2016

from Catalog import Catalog

class CANDELS_EGS_Stefanon_2016(Catalog):

    def __init__(self):
        pass


    def position_in_cat(self,ra,dec): #error assumed to be small and this is approximate anyway
        return True