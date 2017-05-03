#base class for catalogs (essentially an interface class)
#all Catalogs classes must implement:

class Catalog:
    def __init__(self):
        pass

    def position_in_cat(self,ra,dec): #error assumed to be small and this is approximate anyway
        return False    #should be overwritten by child