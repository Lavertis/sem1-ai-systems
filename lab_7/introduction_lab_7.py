from pycaret.anomaly import *
from pycaret.datasets import get_data

seed = 2023
dataset = get_data('anomaly')
data = dataset.sample(frac=0.95, random_state=seed).reset_index(drop=True)
data_unseen = dataset.drop(data.index).reset_index(drop=True)

exp = setup(data, normalize=True)

model = create_model('abod')
labeled_data = assign_model(model)

plot_model(model)

predict_model(model, data=data_unseen)
