from Util import *
import logging
import time
from NetDefinitions import activation_functions
from Config import Config

if __name__ == '__main__':
    log = logging.getLogger('l')
    logging.basicConfig(format='%(asctime)s:  %(message)s')
    log.setLevel(logging.DEBUG)
    # coloredlogs.install(level='DEBUG', fmt='%(asctime)s %(message)s', logger=log) # if i activate this then all the print messages are displayed

    log_name = time.strftime("%Y.%m.%d-%H.%M.%S") + ".log"

    fh = logging.FileHandler(log_name)
    formatter = logging.Formatter('%(asctime)s %(message)s')
    fh.setFormatter(formatter)
    log.addHandler(fh)

    start_time = time.time()

    test = True
    # test = False

    h = activation_functions

    C = Config(log)


    assert True