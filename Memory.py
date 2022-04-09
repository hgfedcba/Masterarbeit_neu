from time import time


# Hier speichere ich verschiedene Ergebnisse eines Durchlaufs für die Ausgabe
class Memory:
    def __init__(self):
        self.start_time = time()
        self.end_time = time()
        self.total_net_durations_per_validation = []  # Wie viel zeit pro validation durch auswertung/training vom netz verwendet wurde. Diese zeit kann ich nicht verringern
        self.pretrain_duration = 0
        self.pretrain_net_duration = 0
        self.single_train_durations = []  # die dauer der einzelnen trainingsiterationen
        self.train_durations_per_validation = []  # für plots ist es nett, sie auch so zu speichern
        self.val_durations = []
        self.test_duration = 0

        self.val_continuous_value_list = []
        self.val_discrete_value_list = []

        self.average_train_payoffs = []
        self.average_pretrain_payoffs = []

        self.average_val_stopping_time = []

        self.net_resets = ""
