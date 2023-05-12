from pycaret.classification import *
from pycaret.datasets import get_data

dataset_list = get_data('index')

dataset = get_data('satellite')
dataset = dataset.sample(frac=0.2, random_state=42).reset_index(drop=True)

# check the shape of data
print(dataset.shape)

data = dataset.sample(frac=0.8, random_state=1).reset_index(drop=True)
data_unseen = dataset.drop(data.index).reset_index(drop=True)

print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions: ' + str(data_unseen.shape))

# Redukcja wymiarów do 50% wyjaśnionej wariancji
exp_clf101_50 = setup(data=data, target='Class', session_id=123, pca=True, pca_components=0.5)
compare_models()

# Redukcja wymiarów do 75% wyjaśnionej wariancji
exp_clf101_75 = setup(data=data, target='Class', session_id=123, pca=True, pca_components=0.75)
compare_models()

# Redukcja wymiarów do 90% wyjaśnionej wariancji
exp_clf101_90 = setup(data=data, target='Class', session_id=123, pca=True, pca_components=0.9)
compare_models()

# Brak redukcji wymiarów
exp_clf101 = setup(data=data, target='Class', session_id=123)
compare_models()
