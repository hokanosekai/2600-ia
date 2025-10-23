import pandas as pd
import numpy as np
import skops.io as skio
from os import cpu_count
import os

# Imports Scikit-learn
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin

# --- NOUVEAUX IMPORTS ---
from sklearn.model_selection import GridSearchCV # L'outil de "bruteforce"
from imblearn.pipeline import Pipeline as ImbPipeline 
# from imblearn.over_sampling import SMOTE # Plus besoin si target=750
from imblearn.under_sampling import RandomUnderSampler

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
        return X_df

# --- 2. Exécution principale du script ---

if __name__ == "__main__":
    
    ORIGINAL_DATA_PATH = 'Oversampling_1500.parquet' 
    TARGET_SAMPLES = 1500 # L'objectif pour chaque classe
    
    print("\n--- Démarrage du Pipeline (avec GridSearchCV) ---")
    print(f"Chargement de {ORIGINAL_DATA_PATH} (version originale)...")
    try:
        df = pd.read_parquet(ORIGINAL_DATA_PATH)
    except FileNotFoundError:
        print(f"ERREUR: Le fichier '{ORIGINAL_DATA_PATH}' n'a pas été trouvé.")
        exit()

    print(f"Données chargées. Forme: {df.shape}")
    print("Distribution originale des classes :")
    print(df['ClassLabel'].value_counts())

    # --- Définition des stratégies d'échantillonnage (CORRIGÉ) ---
    
    # 1. PAS DE SMOTE
    # La plus petite classe (Portscan: 1578) est > TARGET_SAMPLES (750)
    # Nous n'avons donc besoin que de SOUS-échantillonnage.
    
    # 2. Sous-échantillonnage (RandomUnderSampler)
    # On réduit TOUTES les classes à 750.
    all_classes = df['ClassLabel'].unique()
    under_strategy = {cls: TARGET_SAMPLES for cls in all_classes}
    print(f"\nStratégie d'échantillonnage : Ramener toutes les classes à {TARGET_SAMPLES} échantillons.")


    # --- Définition du Pipeline IMBLEARN ---
    pipeline = ImbPipeline([
        ('handle_infinity', InfinityHandler()),
        ('imputer', SimpleImputer(strategy='median')),
        
        # ÉTAPE D'ÉCHANTILLONNAGE :
        ('under_sampler', RandomUnderSampler(sampling_strategy=under_strategy, random_state=42)),
        
        # CLASSIFIEUR :
        ('classifier', RandomForestClassifier(
            class_weight='balanced',# Crucial pour les classes rares
            random_state=42,        # Pour la reproductibilité
            n_jobs=-1               # Utilise tous les cœurs
        ))

    ])
    
    # --- DÉFINITION DE LA GRILLE DE "BRUTEFORCE" ---
    # 'classifier__' fait référence à l'étape 'classifier' du pipeline
    param_grid = {
        'classifier__n_estimators': [100, 200], # Tester 100 ou 200 arbres
        'classifier__max_depth': [15, 30],       # Tester des arbres de profondeur 15 ou 30
        'classifier__min_samples_leaf': [1, 10]  # Tester 1 ou 10 feuilles min
    }
    # Total de combinaisons: 2 * 2 * 2 = 8
    # Augmentez ces listes pour un "bruteforce" plus large
    
    print("\nGrille de paramètres à tester :")
    print(param_grid)

    # --- Préparation des données (Jeu complet) ---
    X = df.drop(columns='ClassLabel', axis=1)
    y = df['ClassLabel']
    
    # --- Initialisation de GridSearchCV ---
    # cv=3 : 3-fold cross-validation. Le script va s'entraîner 8 * 3 = 24 fois
    # scoring='f1_weighted': Le F1-score est souvent meilleur pour le déséquilibre
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=2, 
        scoring='f1_weighted', # Plus robuste que 'accuracy'
        n_jobs=-1, # Utilise tous les coeurs
        verbose=2  # Affiche la progression
    )

    # --- Entraînement (GridSearch remplace le .fit() simple) ---
    print(f"\nDébut du GridSearchCV (test de {8 * 3} modèles)...")
    grid_search.fit(X, y)
    print("GridSearchCV terminé.")

    # --- Affichage des résultats ---
    print("\n--- MEILLEURS RÉSULTATS ---")
    print(f"Meilleur score (F1-Weighted) : {grid_search.best_score_:.4f}")
    print("Meilleurs paramètres trouvés :")
    print(grid_search.best_params_)

    # 'grid_search.best_estimator_' est le pipeline final,
    # entraîné sur 100% des données avec les meilleurs paramètres
    best_model = grid_search.best_estimator_

    # Sauvegarde du modèle
    print("\nSauvegarde du MEILLEUR modèle final sous 'student_model.skio'...")
    with open("student_model.skio", "wb") as f:
        skio.dump(best_model, f)

    print("Modèle sauvegardé avec succès.")
    print("Mission terminée.")