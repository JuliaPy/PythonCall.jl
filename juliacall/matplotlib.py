"""
Minimal matplotlib backend which shows plots using Julia's display mechanism.
"""

from matplotlib.backend_bases import Gcf, FigureManagerBase
from matplotlib.backends.backend_agg import FigureCanvasAgg
from juliacall import Base as jl

FigureCanvas = FigureCanvasAgg

def show(format=None):
    for figmanager in Gcf.get_all_fig_managers():
        figmanager.show(format=format)

FORMAT_TO_MIME = {
    'png': 'image/png',
    'jpg': 'image/jpeg',
    'jpeg': 'image/jpeg',
    'tif': 'image/tiff',
    'tiff': 'image/tiff',
    'svg': 'image/svg+xml',
    'pdf': 'application/pdf',
}

class FigureManager(FigureManagerBase):
    def show(self, format=None):
        fig = self.canvas.figure
        mime = FORMAT_TO_MIME.get(format)
        if format is None:
            jl.display(fig)
        elif mime is None:
            raise ValueError('invalid format {}, expecting one of: {}'.format(format, ', '.join(FORMAT_TO_MIME.keys())))
        else:
            jl.display(mime, fig)
