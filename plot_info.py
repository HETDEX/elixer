import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class Plot_Info():
    def __init__(self, fig_x, fig_y,dpi,grid_x, grid_y):
        self.fig = plt.Figure(figsize=(fig_x, fig_y),dpi=dpi)
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        #plt.gca().axis('off')
        self.gs = gridspec.GridSpec(grid_x, grid_y)
        self.current_row = 0
        self.next_row = 0


    def get_next_row(self,num_rows = 1):
        self.current_row = self.next_row
        self.next_row += num_rows
        return self.current_row




