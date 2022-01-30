import coloredlogs
import logging
import time

import matplotlib
from ConfigInitializer import ConfigInitializer

"""
Eigene Kniffe:
Netz x-K
nutze u statt U
u[N] = 1
trainingsset, aber das bringt vermutlich nichts da overfitting nicht exisitiert
antitheitc variables
input sortieren
"""

# Options:

# Inhalt:
# sklearn -> gradient boost, random forrest
# catboost?
# betrachte das Problem als ein Klassifikationsproblem
# lineare regression etc ausprobieren
# graphikkarte (überraschend schwer)  andere pytorch installation!
# random forest 10.000 sample auf einmal und dann einmal iterieren.

# Grafik:
# plotly express
# colorcode output

# TODO: train with random origin (+/-10%) every 2nd iteration

# TODO: decide on russian parameters

# TODO: Save gif of 2d plot

# VZibglagdbölsgbunrslgubnsrlhsrbnthlsrtghöbhnuhunu

# recall conjectured upper bound

# Note: RNG freeze funktioniert (tested with True/False bei sorted input ohne funktion)

# recall robbins bound from reference also has an explicit realization

# test regularization, dropout      (not functional, breaks alg20)
# <- both are supposed to reduce overfitting, but as overfitting is not possible in my setting it shouldn't help (much)

# TODO: noch einmal gpu (ich muss calculate payoffs anpassen)

# writing
# TODO: pretrainiere mehrere netze gleichzeitig
"""
Aufbau des Programms:
- In der main Methode wird festgelegt, 
"""

if __name__ == '__main__':
    if False:
        import Shortened_RobbinsModel
        from Util import *
        from matplotlib.pyplot import xlabel, ylabel, grid
        import matplotlib.pyplot as plt

        plt.rc('font', size=12)  # controls default text size
        plt.rc('axes', titlesize=12)  # fontsize of the title
        plt.rc('axes', labelsize=12)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=12)  # fontsize of the x tick labels
        plt.rc('ytick', labelsize=12)  # fontsize of the y tick labels
        plt.rc('legend', fontsize=12)  # fontsize of the legend
        n = 19
        s = Shortened_RobbinsModel.Shortened_RobbinsModel(n)
        h = s.getpath_dim()
        # I draw n=0 ... 19 as those are the values used by the net
        draw_connected_points(range(n), s.getpath_dim(), do_scatter=True, plot_number=1)
        draw_connected_points(range(n), np.asarray(range(n))+1, do_scatter=True, plot_number=1)
        plt.legend(["Verringert", "Alle"])
        xlabel("Zeitpunkt")
        ylabel("Eingabedaten")
        grid(True)
        plt.ylim([0, n+1])
        plt.show()

    if False:
        import Shortened_RobbinsModel
        from Util import *
        from matplotlib.pyplot import xlabel, ylabel, grid
        import matplotlib.pyplot as plt
        from NetDefinitions import add_am_put_default_pretrain, pretrain_functions

        K = 100

        add_am_put_default_pretrain(K, 16)

        plt.rc('font', size=12)  # controls default text size
        plt.rc('axes', titlesize=12)  # fontsize of the title
        plt.rc('axes', labelsize=12)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=12)  # fontsize of the x tick labels
        plt.rc('ytick', labelsize=12)  # fontsize of the y tick labels
        plt.rc('legend', fontsize=12)  # fontsize of the legend
        # I draw n=0 ... 19 as those are the values used by the net
        x = np.asarray(range(K-25, K+10))*1.0
        draw_function(x, pretrain_functions[2], linewidth=8)
        """
        y = np.ones_like(x)
        f = pretrain_functions[2]
        for m in range(x.size):
            y[m] = f(x[m])

        draw_connected_points(x, y, plot_number=1)
        """
        # plt.legend(["Verringert", "Alle"])
        xlabel("Stock Price")
        ylabel("Target Value")
        grid(True)
        plt.ylim([0, 1])
        plt.show()

    matplotlib.use("Agg")  # sets a convenient mode for matplotlib

    # configure log
    log = logging.getLogger('l')
    log.setLevel(logging.INFO)
    level_styles = dict(
        spam=dict(color='green', faint=True),
        debug=dict(color='green'),
        verbose=dict(),
        info=dict(color='blue'),
        notice=dict(color='magenta'),
        warning=dict(color='yellow'),
        success=dict(color='green', bold=True),
        error=dict(color='red'),
        critical=dict(color='red', bold=True),
    )
    coloredlogs.install(level='INFO', fmt='%(asctime)s %(message)s', logger=log, level_styles=level_styles)

    log_name = time.strftime("%Y.%m.%d-%H.%M.%S") + ".log"

    fh = logging.FileHandler(log_name)
    formatter = logging.Formatter('%(asctime)s %(message)s')
    fh.setFormatter(formatter)
    log.addHandler(fh)

    '''
    # log color test
    log.debug("this is a debugging message")
    log.info("this is an informational message")
    log.warning("this is a warning message")
    log.error("this is an error message")
    log.critical("this is a critical message")
    '''
    # 4312 = am put, 0 = test, 4411_2, 4411_5, R0, R00, R12, R13, R20, R30, R40 und 0, W, S als pre und s, l, f als suffixe
    log.warning("Start")
    # TODO: R3f crasheds memory, don'T!
    ConfigInitializer(4411_5, log)
    """
    import os
    time.sleep(5)
    os.chdir("../current run2")
    ConfigInitializer("FR12f", log)
    
    time.sleep(5)
    # os.chdir("C:/Users/Olus/Desktop/Masterarbeit Mathe/new computer/current run3")
    os.chdir("../current run3")
    ConfigInitializer("SR20", log)
    
    time.sleep(5)
    os.chdir("../current run4")
    ConfigInitializer("FR12f", log)
    """
    log.warning("The End")
