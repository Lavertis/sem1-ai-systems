from pycaret.classification import *
from pycaret.datasets import get_data

dataset = get_data('credit')

# check the shape of data
print(dataset.shape)

data = dataset.sample(frac=0.95, random_state=786).reset_index(drop=True)
data_unseen = dataset.drop(data.index).reset_index(drop=True)

print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions: ' + str(data_unseen.shape))

exp_clf101 = setup(data=data, target='default', session_id=123)
compare_models()

print("\n####################\nCreating Decision Tree Model\n####################\n")
dt = create_model('dt')
print("\n####################\nCreating K Nearest Neighbors Model\n####################\n")
knn = create_model('knn')
print("\n####################\nCreating Random Forest Model\n####################\n")
rf = create_model('rf')

# trained model object is stored in the variable 'dt'.
print(dt)

print("\n####################\nTuning Decision Tree Model\n####################\n")
tuned_dt = tune_model(dt)
print("\n####################\nTuning K Nearest Neighbors Model\n####################\n")
tuned_knn = tune_model(knn)
print("\n####################\nTuning Random Forest Model\n####################\n")
tuned_rf = tune_model(rf)

# tuned model object is stored in the variable 'tuned_dt'.
print(tuned_dt)

plot_model(tuned_rf, plot='auc')
plot_model(tuned_rf, plot='pr')
plot_model(tuned_rf, plot='feature')
plot_model(tuned_rf, plot='confusion_matrix')

evaluate_model(tuned_rf)
predict_model(tuned_rf)
final_rf = finalize_model(tuned_rf)
# Final Random Forest model parameters for deployment
print(final_rf)

predict_model(final_rf)
unseen_predictions = predict_model(final_rf, data=data_unseen)
print('\n####################\nUnseen Predictions\n####################\n')
print(unseen_predictions.head())

MODEL_NAME = 'Final RF Model 03May2023'
save_model(final_rf, MODEL_NAME)
saved_final_rf = load_model(MODEL_NAME)
new_prediction = predict_model(saved_final_rf, data=data_unseen)
print('\n####################\nNew Predictions using saved Final RF Model\n####################\n')
print(new_prediction.head())
