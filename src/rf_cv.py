import pandas as pd
import numpy as np
import skops.io as skio

# Imports Scikit-learn
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin

# --- NOUVEL IMPORT ---
from sklearn.model_selection import GridSearchCV # L'outil de "bruteforce"

DATA_PATH = 'data/Custom_Balanced.parquet'  # Chemin vers le jeu de données équilibré
OUTPUT_MODEL_PATH = 'models/student_model-rf-cv-4.skio'  # Chemin pour sauvegarder le modèle final

# --- 1. Définition du Transformateur Personnalisé (inchangé) ---
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
    
if __name__ == "__main__":

    # --- 2. Chargement des données (inchangé) ---
    print(f"Chargement de {DATA_PATH}...")
    try:
        df = pd.read_parquet(DATA_PATH)
    except FileNotFoundError:
        print(f"Erreur: Le fichier '{DATA_PATH}' n'a pas été trouvé.")
        exit()

    print(f"Données chargées. Forme: {df.shape}")
    print("Aperçu des classes :")
    print(df['ClassLabel'].value_counts())

    # --- 3. Définition du Pipeline (inchangé) ---
    # Le classifieur sera défini sans paramètres,
    # car GridSearchCV va les injecter.
    pipeline = Pipeline([
        # ('handle_infinity', InfinityHandler()),
        ('imputer', SimpleImputer(strategy='median')),
        ('classifier', RandomForestClassifier(
            class_weight='balanced', # Gardez ceci si vos données sont déséquilibrées
            random_state=42,
            n_jobs=-1
        ))
    ])

    # --- 4. DÉFINITION DE LA GRILLE DE BRUTEFORCE ---
    # 'classifier__' fait référence à l'étape 'classifier' du pipeline
    # AJOUTEZ/MODIFIEZ LES VALEURS ICI POUR TESTER PLUS DE COMBINAISONS
    param_grid = {
        'classifier__n_estimators': [100, 200],    # Nombre d'arbres
        'classifier__max_depth': [15, 30, None],   # Profondeur max des arbres
        'classifier__min_samples_leaf': [1, 5, 10] # Nb min d'échantillons par feuille
    }
    # Total de combinaisons à tester : 2 * 3 * 3 = 18
    # Avec cv=3 (ci-dessous), cela fera 18 * 3 = 54 entraînements.
    print(f"Grille de paramètres à tester : {param_grid}")


    # --- 5. SÉPARATION DES DONNÉES (POUR GRIDSEARCH) ---
    # Nous utilisons l'ensemble des données, GridSearchCV s'occupe
    # de la validation croisée (cv)
    X = df.drop(columns='ClassLabel', axis=1)
    y = df['ClassLabel']
    print(f"Utilisation de l'ensemble des données ({X.shape}) pour GridSearchCV")


    # --- 6. INITIALISATION ET EXÉCUTION DE GRIDSEARCH ---
    # Remplace les étapes 4, 5, 6, 7 de votre script original
    # cv=3 : 3-fold cross-validation. Plus c'est élevé, plus c'est fiable (et long)
    # verbose=2 : Affiche la progression
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=3, 
        scoring='accuracy', # Métrique à optimiser
        n_jobs=-1,          # Utilise tous les cœurs
        verbose=2
    )

    print("\nDébut du 'bruteforce' (GridSearchCV)...")
    grid_search.fit(X, y)
    print("GridSearchCV terminé.")


    # --- 7. RÉSULTATS ET SAUVEGARDE DU MEILLEUR MODÈLE ---
    print("\n--- MEILLEURS RÉSULTATS ---")
    print(f"Meilleur score (Accuracy) trouvé : {grid_search.best_score_:.4f}")
    print("Meilleurs paramètres trouvés :")
    print(grid_search.best_params_)

    # 'best_estimator_' est le pipeline complet,
    # automatiquement ré-entraîné sur TOUTES les données (X, y)
    # avec les meilleurs paramètres trouvés.
    best_model = grid_search.best_estimator_

    # Sauvegarde du modèle
    print(f"\nSauvegarde du MEILLEUR modèle final sous '{OUTPUT_MODEL_PATH}'...")
    with open(OUTPUT_MODEL_PATH, "wb") as f:
        skio.dump(best_model, f)

    print("Modèle sauvegardé avec succès.")
    print("Mission terminée.")