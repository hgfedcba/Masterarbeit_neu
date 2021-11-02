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

# TODO: Alle Plots überprüfen

# TODO: switch names for test and validation set (überprüfen)

# TODO: Save gif of 2d plot

#VZibglagdbölsgbunrslgubnsrlhsrbnthlsrtghöbhnuhunu

# TODO: recall conjectured upper bound

# TODO: Note: RNG freeze funktioniert (tested with True/False bei sorted input ohne funktion)

# TODO: recall robbins bound from reference also has an explicit realization

# TODO: test regularization, dropout      (not functional, breaks alg20)
# TODO: <- both are supposed to reduce overfitting, but as overfitting is not possible in my setting it shouldn't help (much)

# TODO: andere optimierer

# TODO: übergebe nicht alle sortierten werte, sondern nur die letzten (n-2)*ln(n-3-x)/ln(n-3)


if __name__ == '__main__':
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
    # 4312 = am put, 0 = test, 4411_2, 4411_5, R0, R1, R2, R2l, R3, R4, R12, R12l, Russ0, RW0, RW1, RW2, RW3, RW4
    import os
    log.warning("Start")
    # TODO: fix alg20 and not robins
    ConfigInitializer("R0", log)
    """
    os.chdir("../current run2")
    ConfigInitializer("RW4", log)
    
    os.chdir("C:/Users/Olus/Desktop/Masterarbeit Mathe/new computer/current run3")

    ConfigInitializer("Russ111", log)
    """
    log.info("The End")
    """
    from RobbinsModel import RobbinsModel
    R = RobbinsModel(10)
    h = R.generate_paths(5)
    t = np.zeros(10)
    t[0] = 1
    R.getg(t, h[0])

    print(h)


    from catboost_test_file import main
    main()
    """
