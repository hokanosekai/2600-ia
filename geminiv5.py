import pandas as pd
import numpy as np
import skops.io as skio
from os import cpu_count
import os

# Imports Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin

# --- NOUVEAUX IMPORTS ---
from imblearn.pipeline import Pipeline as ImbPipeline 
from imblearn.over_sampling import SMOTE
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
    
    print("\n--- Démarrage du Pipeline (Méthode SMOTE Corrigée) ---")
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
    
    # 1. Sur-échantillonnage (SMOTE)
    # On augmente SEULEMENT les classes qui ont MOINS de 10k échantillons
    # (Portscan: 1578, Webattack: 2096)
    over_strategy = {
        'Portscan': 10000,
        'Webattack': 10000
    }
    # La plus petite classe (Portscan) a 1578 échantillons.
    # k_neighbors doit être plus petit. On prend 5 par défaut (ce qui est < 1578).
    
    # 2. Sous-échantillonnage (RandomUnderSampler)
    # On réduit TOUTES les classes qui ont PLUS de 10k échantillons
    under_strategy = {
        'Benign': 10000,
        'DDoS': 10000,
        'DoS': 10000,
        'Botnet': 10000,
        'Bruteforce': 10000,   # <-- CORRIGÉ : déplacé ici
        'Infiltration': 10000    # <-- CORRIGÉ : déplacé ici
    }

    # --- Définition du Pipeline IMBLEARN ---
    pipeline = ImbPipeline([
        ('handle_infinity', InfinityHandler()),
        ('imputer', SimpleImputer(strategy='median')),
        
        # ÉTAPES D'ÉCHANTILLONNAGE (utilisent les stratégies corrigées)
        ('smote', SMOTE(sampling_strategy=over_strategy, random_state=42, k_neighbors=5)),
        ('under_sampler', RandomUnderSampler(sampling_strategy=under_strategy, random_state=42)),
        
        # CLASSIFIEUR :
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=15,           # <-- RÉDUIRE
            min_samples_leaf=10,    # <-- AUGMENTER (très important)
            class_weight=None, 
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    print("\nPipeline (avec SMOTE corrigé) défini :")
    print(pipeline)

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

    print(f"Taille entraînement: {X_train.shape} (sera équilibré par le pipeline)")
    print(f"Taille test local: {X_test.shape} (ne sera pas touché)")

    # Entraînement pour évaluation locale
    print("\nDébut de l'entraînement (le pipeline applique SMOTE/UnderSampler)...")
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
    pipeline.fit(X, y)
    print("Entraînement final terminé.")

    # Sauvegarde du modèle
    print("\nSauvegarde du modèle final sous 'student_model.skio'...")
    with open("student_model.skio", "wb") as f:
        skio.dump(pipeline, f)

    print("Modèle sauvegardé avec succès.")