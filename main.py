import coloredlogs
import logging
import time

import matplotlib
from Run import Run

"""
Struktur des Programmes:
Das Programm ist aufgeteilt in verschiedene Klassen, von denen man viele thematisch zusammenordnen kann. Diese sind hier aufgeführt.
1. Modelle: Die Folgen Klassen sind alles Kinder der AbstractMathematicalModel Klasse und implementieren eine Art Modell: MarkovBlackScholesModel, RussianOption, RobbinsModel, Shortened_RobbinsModel,
 Filled_RobbinsModel, W_RobbinsModel
    Weiterhin gehören auch ModelDefinitions und ModelInitializer in den Bereich der Modelle. Ersteres ist eine Sammlung an Definitionen, die ich für die Modelle benutze und letzteres wird aufgerufen
     um die Modelle zu initialisieren. Es gibt noch eine Datei NetDefinitions, in der verschiedene Definitionen durchgeführt werden, die dann von den NN-Klassen verwendet werden. 
2. Algoritmusvarianten: Der Kernalgorithmus ist implementiert in der NN-Klasse. Die Klassen Alg10 und Alg20 sind Kinder der NN Klasse, die andere Algorithmusvarianten implementieren. Eine Übersicht
 welcher Algorithmus hinter welcher Zahl steckt gibt es in der Config-Kalsse.
3. Es gibt mehrere Klassen die die Daten eines Durchgangs verwalten. Sie haben die Aufgaben zu speichern was für ein Durchlauf gerade stattfindet (Config), die in der Masterarbeit angesprochenen Rekordergebnisse zu
 speichern (ProminentResults) und die Daten des Durchgangs zu speichern (Memory).
4. Diese Daten werden dann in der Out-Klasse verwendet, um die Ausgabe zu erzeugen.
5. Eine Run-Klasse, in der ich die Konfiguration für den aktuellen Durchlauf setze und die dann alle anderen Klassen aufruft und initialisiert.
6. Es gibt noch eine Sammlung von Hilfsfunktionen in Util.
7. Die Dateien catboost_test_file, NNKatagoInspired und sklearn_test_file werden nicht genutzt. In den Dateien habe ich weitere Ansaätze ausprobiert die aber nie/nicht mehr funktionierten.

Da es so wichtig ist erwähne ich es noch einmal: Welcher Algorithmus hinter welcher Zahl steht sieht man in der Konfig Klasse im alg_dict .
"""

if __name__ == '__main__':

    # erzeugt die Ausgabe der verkürtzten Ausgabedaten
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

    # erzeugt den Plot der amerikanischen Pretrain funktion
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
    # 4312 = am put, 0 = test, 4411_2 = Bermuda mit 2 Aktien, 4411_5 = Bermuda mit 5 Aktien
    # Hier alles in Anführungszeichen -> R0, R00, R12 = Robbins mit 12 ZV, R13, R20 = Robbins mit 20 ZV, R30, R40, Russ1 = Meistgenutztes russisches Setting, Russ11
    # Es gibt 0, W, S als pre und s, l, f als suffixe. 0, W, S geben Varianten vom Robbins Modell an und s, l gibt an ob der Durchlauf schnell oder lang sein soll. f wurde nicht genutzt
    log.warning("Start")

    # Hier wird der Run aufgerufen. Man kann auch mehrere Runs in Folge ausführen
    Run("R00", log)  # fix bug that 4411_5 scheinbar merkwürdige val/test paths hat
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
