import pandas as pd
from src.dataset.dataset import Dataset


class PresentationDuJeuDeDonnees:

    def __init__(self, avc_dataframe:Dataset)->None:
        self.avc_dataframe = avc_dataframe

    def _printer_le_nom_du_dataset(self)->None:
        print("Analyse du jeu de données : AVC Dataset")

    def _printer_probleme_a_resoudre(self)->None:
        print("C'est un problème de classification supervisée")



    def _print_nb_obs(self, avc_dataframe:pd.DataFrame) -> None:
        print(f"Le nombre d'observations est {len(avc_dataframe)}")

    def _print_nb_variables(self, avc_dataframe:pd.DataFrame) -> None:
        print(f"Le nombre de variables est de {len(avc_dataframe.columns)}")


    def _print_variables(self, avc_dataframe: pd.DataFrame) -> None:
        print(f"Nous avons les informations suivantes sur chaque individus {avc_dataframe.columns}")

    def _print_metrique_optimisee(self)->None:
        print("Les métriques utilisées sont l'accuracy, la précision, le recall et le F1-Score. "
              "Nous avons essayer d'optimiser la précision. "
              "Etant donnée le désequilibre dans les personnes ayant effectivement eu un AVC, la maximisation de la précision était notre but. "
              "L'algorithme permet de déclencher l'alarme pour un individu ayant effectivement les symptômes d'un AVC ")

    def presenter_jeu_de_donnees(self, avc_dataframe:pd.DataFrame) -> None:
        self._printer_le_nom_du_dataset()
        self._printer_probleme_a_resoudre()
        self._print_nb_obs(avc_dataframe)
        self._print_nb_variables(avc_dataframe)
        self._print_variables(avc_dataframe)
        self._print_metrique_optimisee()