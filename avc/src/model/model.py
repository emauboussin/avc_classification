
import sklearn.ensemble
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from src.dataset.dataset import Dataset


class Model:
    def __init__(self, cleaned_data_set: Dataset,number_of_trees: int, train_test_proportion: float) -> None:
        self.number_of_trees = number_of_trees
        self.train_test_proportion = train_test_proportion
        self.cleaned_data_set = cleaned_data_set

    def fit(self)-> sklearn.ensemble.RandomForestClassifier:
        x_train,x_test, y_train, y_test = train_test_split(self.cleaned_data_set.drop("stroke", axis=1),
                                                            self.cleaned_data_set["stroke"],
                                                            stratify=self.cleaned_data_set["stroke"], test_size=1-self.train_test_proportion,
                                                            random_state=42)
        random_forest_classifier = RandomForestClassifier(n_estimators=self.number_of_trees, oob_score=True, random_state=42)
        return random_forest_classifier.fit(x_train, y_train)

    def metrics(self):
        x_train, x_test, y_train, y_test = train_test_split(self.cleaned_data_set.drop("stroke", axis=1),
                                                            self.cleaned_data_set["stroke"],
                                                            stratify=self.cleaned_data_set["stroke"], test_size=1-self.train_test_proportion,
                                                            random_state=42)
        random_forest_classifier = RandomForestClassifier(n_estimators=self.number_of_trees, oob_score=True, random_state=42)
        random_forest_classifier.fit(x_train, y_train)
        predicted = random_forest_classifier.predict(x_test)
        accuracy = accuracy_score(y_test, predicted)
        print(f'Out-of-bag score estimate: {random_forest_classifier.oob_score_:.3}')
        print(f'Mean accuracy score: {accuracy:.3}')

        y_pred = random_forest_classifier.predict(x_test)


        print('Precision: %.3f' % precision_score(y_test, y_pred))
        print('Recall: %.3f' % recall_score(y_test, y_pred))
        print('F1 Score: %.3f' % f1_score(y_test, y_pred))
