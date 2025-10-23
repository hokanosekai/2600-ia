import pandas as pd
import numpy as np
import skops.io as skio
from os import cpu_count
import os

# Imports Scikit-learn
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin

# --- NOUVEAUX IMPORTS ---
from imblearn.pipeline import Pipeline as ImbPipeline 
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
    
    ORIGINAL_DATA_PATH = 'Training.parquet' 
    
    print("\n--- Démarrage du Pipeline (Méthode 50/50 Imblearn) ---")
    print(f"Chargement de {ORIGINAL_DATA_PATH} (version originale)...")
    try:
        df = pd.read_parquet(ORIGINAL_DATA_PATH)
    except FileNotFoundError:
        print(f"ERREUR: Le fichier '{ORIGINAL_DATA_PATH}' n'a pas été trouvé.")
        exit()

    print(f"Données chargées. Forme: {df.shape}")

    # Séparation des données (sur le jeu ORIGINAL)
    print("\nSéparation des données (sur jeu original)...")
    X = df.drop(columns='ClassLabel', axis=1)
    y = df['ClassLabel']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    print(f"Taille entraînement: {X_train.shape}")
    print(f"Taille test local: {X_test.shape}")
    
    # --- Définition de la stratégie d'échantillonnage dynamique ---
    # Calcule le nombre d'attaques DANS LE JEU D'ENTRAÎNEMENT
    n_attacks_train = (y_train != 'Benign').sum()
    print(f"Le jeu d'entraînement contient {n_attacks_train} échantillons d'attaque.")
    print(f"Le pipeline va réduire 'Benign' à {n_attacks_train} échantillons.")
    
    # Stratégie : ne touche à rien, sauf à 'Benign' qu'on réduit à n_attacks_train
    under_strategy_50_50 = {'Benign': n_attacks_train}

    # --- Définition du Pipeline IMBLEARN ---
    pipeline = ImbPipeline([
        ('handle_infinity', InfinityHandler()),
        ('imputer', SimpleImputer(strategy='median')),
        
        # ÉTAPE D'ÉCHANTILLONNAGE :
        ('under_sampler', RandomUnderSampler(
            sampling_strategy=under_strategy_50_50, 
            random_state=42
        )),
        
        # CLASSIFIEUR :
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_leaf=10,
            class_weight=None, # Inutile, on a équilibré à 50/50
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    print("\nPipeline (avec 50/50 UnderSampler) défini.")

    # Entraînement pour évaluation locale
    print("\nDébut de l'entraînement (le pipeline applique l'UnderSampler)...")
    pipeline.fit(X_train, y_train)
    print("Entraînement local terminé.")

    # Évaluation locale
    print("\nÉvaluation sur le jeu de test local (PROPRE)...")
    y_pred = pipeline.predict(X_test)
    local_score = accuracy_score(y_test, y_pred)

    print(f"Score (Accuracy) local : {local_score:.4f}")
    print("\nRapport de classification local (RÉALISTE) :")
    print(classification_report(y_test, y_pred))

    # Entraînement final sur TOUTES les données
    print("\nDébut de l'entraînement final (sur 100% des données)...")
    # Nous devons recalculer la stratégie pour les 100% des données
    n_attacks_full = (y != 'Benign').sum()
    final_strategy = {'Benign': n_attacks_full}
    
    # On re-définit le paramètre du pipeline avant l'entraînement final
    pipeline.set_params(under_sampler__sampling_strategy=final_strategy)
    
    print(f"Entraînement final : réduction de 'Benign' à {n_attacks_full} échantillons.")
    pipeline.fit(X, y)
    print("Entraînement final terminé.")

    # Sauvegarde du modèle
    print("\nSauvegarde du modèle final sous 'student_model.skio'...")
    with open("student_last_model.skio", "wb") as f:
        skio.dump(pipeline, f)

    print("Modèle sauvegardé avec succès.")