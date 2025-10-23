import pandas as pd
import numpy as np
import skops.io as skio
from os import cpu_count
import os

# Imports Scikit-learn
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# --- NOUVEAUX IMPORTS ---
from sklearn.model_selection import GridSearchCV, train_test_split
# L'un des meilleurs modèles pour données tabulaires : rapide et performant
from sklearn.ensemble import HistGradientBoostingClassifier

# --- 1. Transformateur Personnalisé (inchangé) ---
class InfinityHandler(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X_df = pd.DataFrame(X)
        else:
            X_df = X.copy()
        X_df.replace([np.inf], np.finfo(np.float64).max, inplace=True)
        X_df.replace([-np.inf], np.finfo(np.float64).min, inplace=True)
        # HistGradientBoosting gère les NaN nativement, pas besoin d'Imputer
        return X_df

# --- 2. Chargement des données ---
print("Chargement de Training.parquet (Le fichier ORIGINAL)...")
try:
    df = pd.read_parquet("Training.parquet") 
except FileNotFoundError:
    print("Erreur: Le fichier 'Training.parquet' original n'a pas été trouvé.")
    exit()

print(f"Données originales chargées. Forme: {df.shape}")

# --- 3. PRÉPARATION DES DONNÉES ---
X_full = df.drop(columns='ClassLabel', axis=1)
y_full = df['ClassLabel']

# CRÉER UN PETIT ÉCHANTILLON POUR GRIDSEARCH (ex: 5%)
print(f"Création d'un échantillon (5%) pour GridSearchCV...")
# Nous prenons 5% pour le "bruteforce"
# 'stratify=y' est crucial pour garder les classes rares dans l'échantillon
X_search, _, y_search, _ = train_test_split(
    X_full, y_full, 
    train_size=0.05, # <-- On ne prend que 5% des données
    random_state=42, 
    stratify=y_full
)
print(f"Taille de l'échantillon pour GridSearchCV : {X_search.shape}")
del df # Libère la mémoire du DataFrame complet

# --- 4. Définition du Pipeline ---
# Ce pipeline est plus simple et plus rapide
pipeline = Pipeline([
    ('handle_infinity', InfinityHandler()),
    # Pas besoin d'Imputer, HistGradientBoosting gère les NaN
    ('classifier', HistGradientBoostingClassifier(
        random_state=42,
        class_weight='balanced' # <-- Gère le déséquilibre SANS fuite de données
    ))
])

# --- 5. DÉFINITION DE LA GRILLE (Allégée) ---
# 'learning_rate' et 'max_leaf_nodes' sont les clés pour ce modèle
param_grid = {
    'classifier__learning_rate': [0.1, 0.05],     # Taux d'apprentissage
    'classifier__max_leaf_nodes': [31, 50],       # Similaire à max_depth
    'classifier__max_iter': [100, 200]            # Similaire à n_estimators
}
# Total de combinaisons : 2 * 2 * 2 = 8
print(f"\nGrille de paramètres à tester : {param_grid}")

# --- 6. INITIALISATION ET EXÉCUTION DE GRIDSEARCH ---
grid_search = GridSearchCV(
    pipeline,
    param_grid, 
    cv=2,              # 2 folds, plus léger pour votre machine
    scoring='f1_weighted', # Métrique robuste pour le déséquilibre
    n_jobs=-1,         
    verbose=2
)

print("\nDébut du 'bruteforce' (GridSearchCV) sur l'échantillon de 5%...")
# ON ENTRAÎNE SUR LE PETIT ÉCHANTILLON
grid_search.fit(X_search, y_search)
print("GridSearchCV terminé.")

# --- 7. RÉSULTATS ---
print("\n--- MEILLEURS RÉSULTATS (basés sur l'échantillon) ---")
print(f"Meilleur score (F1-Weighted) trouvé : {grid_search.best_score_:.4f}")
print("Meilleurs paramètres trouvés :")
print(grid_search.best_params_)

# --- 8. ENTRAÎNEMENT DU MODÈLE FINAL ---
print("\n--- Entraînement du modèle final ---")
print("Création du pipeline final avec les meilleurs paramètres...")

# On récupère les meilleurs paramètres trouvés
final_params = grid_search.best_estimator_.get_params()

# On crée un NOUVEAU pipeline final
final_pipeline = Pipeline([
    ('handle_infinity', InfinityHandler()),
    ('classifier', HistGradientBoostingClassifier(
        random_state=42,
        class_weight='balanced'
    ))
])

# On applique les meilleurs paramètres
final_pipeline.set_params(**final_params)

print("Entraînement du modèle final sur 100% DES DONNÉES (cela peut prendre du temps)...")
# On entraîne sur TOUTES les données
final_pipeline.fit(X_full, y_full)
print("Entraînement final terminé.")

# --- 9. Sauvegarde du modèle ---
print("\nSauvegarde du MEILLEUR modèle final sous 'student_model.skio'...")
with open("student_model_2.skio", "wb") as f:
    skio.dump(final_pipeline, f)

print("Modèle sauvegardé avec succès.")
print("Mission terminée.")