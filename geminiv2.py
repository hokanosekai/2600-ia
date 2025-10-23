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

# Imports pour le transformateur personnalisé
from sklearn.base import BaseEstimator, TransformerMixin

# --- 1. Fonction pour créer le jeu de données personnalisé ---

def create_custom_balanced_set(original_parquet_path, 
                               output_path="Custom_Balanced.parquet", 
                               random_state=42):
    """
    Crée un jeu de données où:
    - Benign = 7500 échantillons
    - Toutes les autres classes = 1500 échantillons
    """
    print(f"--- Étape de pré-traitement : Création de {output_path} ---")
    
    if not os.path.exists(original_parquet_path):
        print(f"ERREUR: Le fichier parquet d'origine '{original_parquet_path}' est introuvable.")
        return False

    print(f"Chargement du jeu de données complet depuis : {original_parquet_path}...")
    df_full = pd.read_parquet(original_parquet_path)
    
    if 'Label' in df_full.columns:
        df_full = df_full.drop(columns='Label')

    classes = df_full['ClassLabel'].unique()
    print(f"Classes trouvées : {classes}")
    
    sampled_dfs = []
    
    for cls in classes:
        # --- LOGIQUE MODIFIÉE ICI ---
        # 1. Définir le nombre d'échantillons cible
        if cls == 'Benign':
            target_samples = 7500
        else:
            target_samples = 1500
        # --- FIN DE LA MODIFICATION ---
            
        print(f"Traitement de la classe '{cls}' (objectif: {target_samples} échantillons)...")
        class_df = df_full[df_full['ClassLabel'] == cls]
        
        # 2. Vérifier si on a assez de données (on l'a, mais c'est une bonne pratique)
        if len(class_df) < target_samples:
            # Ce cas ne devrait pas arriver avec vos chiffres
            print(f"  Alerte : La classe '{cls}' n'a que {len(class_df)} échantillons.")
            print(f"  Sur-échantillonnage (avec remplacement) pour atteindre {target_samples}...")
            sampled_dfs.append(class_df.sample(n=target_samples, random_state=random_state, replace=True))
        else:
            # SOUS-ÉCHANTILLONNAGE (le seul cas qui s'appliquera ici)
            print(f"  La classe '{cls}' a {len(class_df)} échantillons.")
            print(f"  Sous-échantillonnage (sans remplacement) pour atteindre {target_samples}...")
            sampled_dfs.append(class_df.sample(n=target_samples, random_state=random_state, replace=False))
            
    print("\nCombinaison et mélange des échantillons...")
    df_balanced = pd.concat(sampled_dfs)
    
    # Mélange final
    df_balanced = df_balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    print("\n--- Statistiques du nouveau 'Training.parquet' ---")
    print(f"Forme finale : {df_balanced.shape}")
    print(df_balanced['ClassLabel'].value_counts())
    
    print(f"\nSauvegarde dans le fichier : {output_path}...")
    df_balanced.to_parquet(output_path, index=False)
    print("Sauvegarde terminée.")
    print("-" * 50)
    return True


# --- 2. Définition du Transformateur Personnalisé (inchangé) ---
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

# --- 3. Exécution principale du script ---

if __name__ == "__main__":
    
    ORIGINAL_DATA_PATH = 'Training.parquet'
    NEW_TRAINING_PATH = 'Custom_Balanced.parquet' # Nouveau nom

    # Exécute la fonction de création personnalisée
    success = create_custom_balanced_set(
        original_parquet_path=ORIGINAL_DATA_PATH,
        output_path=NEW_TRAINING_PATH
    )

    if not success:
        print("Arrêt du script car le fichier d'origine n'a pas pu être traité.")
        exit()

    # --- ÉTAPE B : Pipeline du Challenge (utilise le fichier créé) ---
    
    print(f"\n--- Démarrage du Pipeline sur {NEW_TRAINING_PATH} ---")
    df = pd.read_parquet(NEW_TRAINING_PATH)
    print(f"Données chargées. Forme: {df.shape}")

    # Définition du Pipeline
    pipeline = Pipeline([
        ('handle_infinity', InfinityHandler()),
        ('imputer', SimpleImputer(strategy='median')),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_leaf=10,
            # Le jeu est déséquilibré (7500 vs 1500), donc 'balanced' est une bonne idée
            class_weight='balanced', 
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    print("\nPipeline défini :")
    print(pipeline)

    X = df.drop(columns='ClassLabel', axis=1)
    y = df['ClassLabel']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y # Important pour garder l'équilibre 7500/1500 dans le test
    )

    print(f"Taille entraînement: {X_train.shape}")
    print(f"Taille test local: {X_test.shape}")

    # Entraînement pour évaluation locale
    print("\nDébut de l'entraînement (évaluation locale)...")
    pipeline.fit(X_train, y_train)
    print("Entraînement local terminé.")

    # Évaluation locale
    print("\nÉvaluation sur le jeu de test local (Score fiable)...")
    y_pred = pipeline.predict(X_test)
    local_score = accuracy_score(y_test, y_pred)

    print(f"Score (Accuracy) local : {local_score:.4f}")
    print("\nRapport de classification local :")
    print(classification_report(y_test, y_pred))

    # Entraînement final sur TOUTES les données équilibrées
    print("\nDébut de l'entraînement final (sur 100% des données)...")
    pipeline.fit(X, y)
    print("Entraînement final terminé.")

    # Sauvegarde du modèle
    print("\nSauvegarde du modèle final sous 'student_model.skio'...")
    with open("student_model.skio", "wb") as f:
        skio.dump(pipeline, f)

    print("Modèle sauvegardé avec succès.")
    print("Mission terminée.")