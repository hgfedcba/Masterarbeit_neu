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

#VZibglagdbölsgbunrslgubnsrlhsrbnthlsrtghöbhnuhunu

# TODO: recall conjectured upper bound

# TODO: Note: RNG freeze funktioniert (tested with True/False bei sorted input ohne funktion)

# TODO: recall robbins bound from reference also has an explicit realization

# TODO: test regularization, dropout      (not functional, breaks alg20)
# TODO: <- both are supposed to reduce overfitting, but as overfitting is not possible in my setting it shouldn't help (much)

# TODO: noch einmal gpu (ich muss calculate payoffs anpassen)

# TODO: fix alg20, 14 and not robins

# TODO: alg 2 robbins (without sorting)   3 versionen, einmal nullen vorne, einmal nullen hinten und einmal einsen hinten, immer mit extra t

# TODO: pretrainiere mehrere netze gleichzeitig

# TODO: erwähne die dimension auch bei alg0 zu übergeben


if __name__ == '__main__':
    if False:
        import Shortened_RobbinsModel
        from Util import *
        n = 39
        s = Shortened_RobbinsModel.Shortened_RobbinsModel(n)
        h = s.getpath_dim()
        # I draw n=0 ... 19 as those are the values used by the net
        draw_connected_points(range(n), s.getpath_dim(), do_scatter=True, plot_number=1)
        draw_connected_points(range(n), np.asarray(range(n))+1, do_scatter=True, plot_number=1)
        plt.legend(["shortened", "full"])
        xlabel("net number")
        ylabel("number of arguments")
        grid(True)
        plt.ylim([0, n+1])
        plt.show()

    matplotlib.use("Agg")

    log = logging.getLogger('l')
    # logging.basicConfig(format='%(asctime)s:  %(message)s')
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
    log.debug("this is a debugging message")
    log.info("this is an informational message")
    log.warning("this is a warning message")
    log.error("this is an error message")
    log.critical("this is a critical message")
    '''
    # 4312 = am put, 0 = test, 4411_2, 4411_5, R0, R00, R12, R13, R20, R30, R40 und 0, W, S als pre und s, l, f als suffixe  # TODO: mache lokale pfade für alles
    import os
    log.warning("Start")
    # TODO: R3f crasheds memory, don'T!
    # TODO: lade testpfade erst in test und überschreibe dann mit None
    ConfigInitializer("R30", log)
    """
    time.sleep(5)
    os.chdir("../current run2")
    ConfigInitializer("SR40", log)

    time.sleep(5)
    # os.chdir("C:/Users/Olus/Desktop/Masterarbeit Mathe/new computer/current run3")
    os.chdir("../current run3")
    ConfigInitializer("WR40", log)

    time.sleep(5)
    os.chdir("../current run4")
    ConfigInitializer("FR40", log)
    """
    log.warning("The End")
    """
    from catboost_test_file import main
    main()
    """
