import global_config as G
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches


class Plot_Info():
    def __init__(self, fig_x, fig_y,dpi,grid_x, grid_y):
        self.fig = plt.Figure(figsize=(fig_x, fig_y),dpi=dpi)
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        #plt.gca().axis('off')
        self.gs = gridspec.GridSpec(grid_x, grid_y)


