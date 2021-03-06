import coloredlogs
import logging
import time

import numpy as np

from ConfigInitializer import ConfigInitializer

"""
Eigene Kniffe:
Netz x-K
nutze u statt U
u[N] = 1
trainingsset, aber das bringt vermutlich nichts da overfitting nicht exisitiert
antitheitc variables

"""

# TODO:!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# TODO: experimentiere herum mit bel startwerten bei den trainingspfaden.
# TODO:!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

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


"""
1. implemntiere robbins klasse
2. abstrahiere aus mathematical model und robbins problem ein abstract mathematical model
3. kriege den alten algorithmus zum laufen mit dem abstarct mathematical model
4. kriege alg0 zum laufen mit dem robbins problem
5. implementiere neuen algorithmus (lerne die netze nacheinander von hinten nach vorne)
"""

import torch


if __name__ == '__main__':
    import matplotlib
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
    # 4312 = am put, 0 = test, 4411_2, 4411_5, R0, R1, R2, R3, Russ0
    CI = ConfigInitializer("Russ111", log)

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
