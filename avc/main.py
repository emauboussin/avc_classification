import argparse

from src.presentation_du_jeu_de_donnees.presentation_du_jeu_de_donnees import PresentationDuJeuDeDonnees
from src.model.model import Model
from src.dataset.dataset import Dataset

if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--path_to_data", type=str, required=True)
    argument_parser.add_argument("--number_of_trees", type=int, required=True)
    argument_parser.add_argument("--train_test_proportion", type=float, required=True)

    arguments = argument_parser.parse_args()

    avc_dataframe = Dataset(arguments.path_to_data)
    presentation_jeu_de_donnes = PresentationDuJeuDeDonnees(avc_dataframe)
    presentation_jeu_de_donnes.presenter_jeu_de_donnees(avc_dataframe.raw_data)

    cleaned_dataframe = avc_dataframe.preprocessing()

    model = Model(cleaned_dataframe, arguments.number_of_trees, arguments.train_test_proportion)
    modele_fitte = model.metrics()

    # def main(path_to_data: str, number_of_trees: int, train_test_proportion: float) -> None:
    #     presentation_jeu_de_donnes = PresentationDuJeuDeDonnees()
    #
    #     raw_data_set = Dataset(path_to_data)
    #     cleaned_data_set = raw_data_set.transformation()
    #
    #     training_model = Model(cleaned_data_set, number_of_trees, train_test_proportion)
    #     suv = SportUtilityVehicle(car_brand, year_of_construction, car_power)
    #     technical_car_controller.start_and_control_car(suv)
    #
    #     tampered_suv = SportUtilityVehicle("Toyota", 2020, 200)
    #     car_unlocker.tampered_car(tampered_suv, 2022, 500)
    #     technical_car_controller.start_and_control_car(tampered_suv)
