from pycaret.classification import *
from pycaret.datasets import get_data

# Wykorzystując bibliotekę PyCaret przeprowadź klasyfikację zbioru danych
# ‘heart_disease.
# a. Do zbioru danych modelu wybierz 99% danych, a generator liczb losowych
# zainicjuj liczbą będącą trzema ostatnimi cyframi Twojego numeru
# albumu/indexu. Pozostałe dane zapisz w zbiorze niedostępnym dla modelu
# w trakcie uczenia (unseen). Do raportu proszę zapisać wymiary otrzymanych
# zbiorów danych.
# b. Przeprowadź klasyfikację używając wszystkich dostępnych modeli, tuning
# dwóch najlepszych modeli i ich finalizację używając całego zbioru do
# modelowania. Do raportu zapisz otrzymane wyniki metryk dla wszystkich
# modeli, wybrane 2 najlepsze modele, parametry modeli po tuningu, wyniki
# dla zbioru unseen, parametry modeli po finalizacji i wyniki dla zbioru unseen
# po finalizacji.
# c. Przeprowadź redukcję wymiarów używając PCA z zachowaniem 80%
# wyjaśnionej wariancji i powtórz klasyfikację z podpunktu b. W raporcie poza
# wynikami analogicznymi to tych z podpunktu b. wpisz liczbę
# pozostawionych wymiarów

seed = 920
FOLD_COUNT = 3
ITER_COUNT = 5

dataset = get_data('heart_disease')
data = dataset.sample(frac=0.99, random_state=seed).reset_index(drop=True)  # (267, 14)
data_unseen = dataset.drop(data.index).reset_index(drop=True)  # (3, 14)

print("\n####################\nDimensionality Reduction with PCA\n####################\n")
clf_pca = setup(data=data, target='Disease', fold=FOLD_COUNT, session_id=seed, pca=True, pca_components=0.8)

print("\n####################\nComparing Models after PCA\n####################\n")
best_models = compare_models(n_select=2, fold=FOLD_COUNT)

print("\n####################\nTuning Models after PCA\n####################\n")
tuned_models = [tune_model(model, fold=FOLD_COUNT, n_iter=ITER_COUNT) for model in best_models]

print("\n####################\nTuned Model parameters\n####################\n")
for model in tuned_models:
    print(model)

print("\n####################\nPredictions for Unseen Data after Tuning\n####################\n")
predictions_after_tuning = [predict_model(model, data=data_unseen) for model in tuned_models]

print("\n####################\nFinalizing Models after PCA\n####################\n")
final_models = [finalize_model(model) for model in tuned_models]

print("\n####################\nFinalized Model parameters\n####################\n")
for model in final_models:
    print(model)

print("\n####################\nPredictions for Unseen Data after Finalizing\n####################\n")
predictions_after_finalizing = [predict_model(model, data=data_unseen) for model in final_models]
