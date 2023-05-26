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

print("\n####################\nCreating Model\n####################\n")
clf = setup(data=data, target='Class', fold=3, session_id=seed)

print("\n####################\nComparing Models\n####################\n")
best_models = compare_models(n_select=2, fold=3)

print("\n####################\nTuning Models\n####################\n")
tuned_1 = tune_model(best_models[0], fold=3)
tuned_2 = tune_model(best_models[1], fold=3)

print("\n####################\nTuned Model parameters\n####################\n")
print(tuned_1)
print(tuned_2)

print("\n####################\nPredicting Models after tuning\n####################\n")
predictions_after_tuning_1 = predict_model(tuned_1, data=data_unseen)
predictions_after_tuning_2 = predict_model(tuned_2, data=data_unseen)

print("\n####################\nFinalizing Models\n####################\n")
final_1 = finalize_model(tuned_1)
final_2 = finalize_model(tuned_2)

print("\n####################\nFinalized Model parameters\n####################\n")
print(final_1)
print(final_2)

print("\n####################\nPredicting Models after Finalizing\n####################\n")
predictions_after_finalizing_1 = predict_model(final_1, data=data_unseen)
predictions_after_finalizing_2 = predict_model(final_2, data=data_unseen)
