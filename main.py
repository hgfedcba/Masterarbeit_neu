from Util import *
import logging
import time
from NetDefinitions import activation_functions
from Config import Config
from ConfigInitializer import ConfigInitializer
from Memory import Memory

"""
Eigene Kniffe:
Netz x-K
nutze u statt U
u[N] = 1
trainingsset, aber das bringt vermutlich nichts da overfitting nicht exisitiert
antitheitc variables

"""

if __name__ == '__main__':
    log = logging.getLogger('l')
    logging.basicConfig(format='%(asctime)s:  %(message)s')
    log.setLevel(logging.INFO)
    # coloredlogs.install(level='DEBUG', fmt='%(asctime)s %(message)s', logger=log) # if i activate this then all the print messages are displayed

    log_name = time.strftime("%Y.%m.%d-%H.%M.%S") + ".log"

    fh = logging.FileHandler(log_name)
    formatter = logging.Formatter('%(asctime)s %(message)s')
    fh.setFormatter(formatter)
    log.addHandler(fh)

    # 4312 = am put, 0 = test
    CI = ConfigInitializer(1, log)

    log.info("The End")

