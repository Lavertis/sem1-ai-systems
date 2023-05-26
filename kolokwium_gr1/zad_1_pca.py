from pycaret.classification import *
from pycaret.datasets import get_data

# Wykorzystując bibliotekę PyCaret przeprowadź klasyfikację zbioru danych ‘cancer’.
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
# c. Przeprowadź redukcję wymiarów używając PCA z zachowaniem 75%
# wyjaśnionej wariancji i powtórz klasyfikację z podpunktu b. W raporcie poza
# wynikami analogicznymi to tych z podpunktu b. wpisz liczbę
# pozostawionych wymiarów.

seed = 920

dataset = get_data('cancer')
data = dataset.sample(frac=0.99, random_state=seed).reset_index(drop=True)  # (676, 10)
data_unseen = dataset.drop(data.index).reset_index(drop=True)  # (7, 10)

print("\n####################\nDimensionality Reduction with PCA\n####################\n")
clf_pca = setup(data=data, target='Class', fold=3, session_id=seed, pca=True, pca_components=0.75)
# zostały pozostawione 3 składowe główne

print("\n####################\nComparing Models after PCA\n####################\n")
best_models = compare_models(n_select=2, fold=3)

print("\n####################\nTuning Models after PCA\n####################\n")
tuned_models = [tune_model(model, fold=3) for model in best_models]

print("\n####################\nTuned Model parameters\n####################\n")
for model in tuned_models:
    print(model)

print("\n####################\nPredictions for Unseen Data after tuning\n####################\n")
predictions_after_tuning = [predict_model(model, data=data_unseen) for model in tuned_models]

print("\n####################\nFinalizing Models after PCA\n####################\n")
final_models = [finalize_model(model) for model in tuned_models]

print("\n####################\nFinalized Model parameters\n####################\n")
for model in final_models:
    print(model)

print("\n####################\nPredictions for Unseen Data after PCA\n####################\n")
predictions_after_finalizing = [predict_model(model, data=data_unseen) for model in final_models]
