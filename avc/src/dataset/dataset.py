import pandas as pd
from sklearn import preprocessing

GENDER = "gender"


class Dataset:

    def __init__(self, path_to_data: str) -> None:
        self.raw_data = pd.read_csv(path_to_data)

    def _transformation(self) -> pd.DataFrame:
        raw_avc_data = self.raw_data[(self.raw_data.gender != 'Other')]

        raw_avc_data = self._drop_id(raw_avc_data)
        one_hot_encoded_data = pd.get_dummies(raw_avc_data, columns=['work_type', 'Residence_type'])

        # label_encoder object knows how to understand word labels.
        label_encoder = preprocessing.LabelEncoder()
        # Encode labels in column 'species'.
        one_hot_encoded_data[GENDER] = label_encoder.fit_transform(one_hot_encoded_data[GENDER])
        one_hot_encoded_data['ever_married'] = label_encoder.fit_transform(one_hot_encoded_data['ever_married'])
        one_hot_encoded_data['smoking_status'] = label_encoder.fit_transform(one_hot_encoded_data['smoking_status'])

        one_hot_encoded_data.bmi.fillna(one_hot_encoded_data.bmi.median(), inplace=True)

        one_hot_encoded_data['age'] = pd.cut(x=one_hot_encoded_data['age'], bins=[0, 25, 45, 65, 85])
        one_hot_encoded_data['age'] = label_encoder.fit_transform(one_hot_encoded_data['age'])
        one_hot_encoded_data['bmi'] = pd.cut(x=one_hot_encoded_data['bmi'], bins=[0, 18.5, 25, 30, 100])
        one_hot_encoded_data['bmi'] = label_encoder.fit_transform(one_hot_encoded_data['bmi'])

        return one_hot_encoded_data

    def _drop_id(self, raw_avc_data):
        raw_avc_data = raw_avc_data.drop('id', axis=1)
        return raw_avc_data

    def preprocessing(self) -> pd.DataFrame():
        cleaned_data = self._transformation()
        return cleaned_data
