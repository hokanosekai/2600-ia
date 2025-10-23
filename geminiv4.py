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

# --- 1. Fonction pour créer le jeu de données équilibré ---

def create_balanced_training_set(original_parquet_path, 
                                 output_path="Training.parquet", 
                                 samples_per_class=1500, #<- Valeur par défaut changée pour info
                                 random_state=42):
    """
    Crée un jeu de données d'entraînement équilibré en échantillonnant
    le fichier parquet d'origine.
    
    Gère automatiquement le sous-échantillonnage (si classe > N)
    et le sur-échantillonnage (si classe < N).
    """
    print(f"--- Étape de pré-traitement : Création de {output_path} ---")
    
    if not os.path.exists(original_parquet_path):
        print(f"ERREUR: Le fichier parquet d'origine '{original_parquet_path}' est introuvable.")
        print("Veuillez mettre à jour la variable 'ORIGINAL_DATA_PATH' avec le bon chemin.")
        return False

    print(f"Chargement du jeu de données complet depuis : {original_parquet_path}...")
    df_full = pd.read_parquet(original_parquet_path)
    
    if 'Label' in df_full.columns:
        df_full = df_full.drop(columns='Label')

    classes = df_full['ClassLabel'].unique()
    print(f"Classes trouvées : {classes}")
    
    sampled_dfs = []
    
    for cls in classes:
        print(f"Traitement de la classe '{cls}' (objectif: {samples_per_class} échantillons)...")
        class_df = df_full[df_full['ClassLabel'] == cls]
        
        # Cette condition gère TOUT :
        if len(class_df) < samples_per_class:
            # --- SUR-ÉCHANTILLONNAGE (Oversampling) ---
            print(f"  Alerte : La classe '{cls}' n'a que {len(class_df)} échantillons.")
            print(f"  Sur-échantillonnage (avec remplacement) pour atteindre {samples_per_class}...")
            # 'replace=True' permet de ré-utiliser les mêmes lignes
            sampled_dfs.append(class_df.sample(n=samples_per_class, random_state=random_state, replace=True))
        else:
            # --- SOUS-ÉCHANTILLONNAGE (Undersampling) ---
            print(f"  La classe '{cls}' a {len(class_df)} échantillons.")
            print(f"  Sous-échantillonnage (sans remplacement) pour atteindre {samples_per_class}...")
            # 'replace=False' (par défaut) garantit des lignes uniques
            sampled_dfs.append(class_df.sample(n=samples_per_class, random_state=random_state, replace=False))
            
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


# --- 2. Définition du Transformateur Personnalisé (pour le pipeline) ---
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
    
    # --- ÉTAPE A : Création du fichier 'Training.parquet' équilibré ---
    
    # !! MODIFIEZ CE CHEMIN vers votre fichier d'origine !!
    ORIGINAL_DATA_PATH = 'Training.parquet'
    NEW_TRAINING_PATH = 'Oversampling.parquet'

    # Exécute la fonction de création
    success = create_balanced_training_set(
        original_parquet_path=ORIGINAL_DATA_PATH,
        output_path=NEW_TRAINING_PATH,
        samples_per_class=10000  # <--- SEULE MODIFICATION NÉCESSAIRE
    )

    if not success:
        print("Arrêt du script car le fichier d'origine n'a pas pu être traité.")
        exit()

    # --- ÉTAPE B : Pipeline du Challenge (utilise le fichier créé) ---
    
    print("\n--- Démarrage du Pipeline pour le Challenge ---")
    print(f"Chargement de {NEW_TRAINING_PATH} (version 10k/classe)...")
    try:
        df = pd.read_parquet(NEW_TRAINING_PATH)
    except FileNotFoundError:
        print(f"ERREUR: Le fichier '{NEW_TRAINING_PATH}' n'a pas été trouvé ou créé.")
        exit()

    print(f"Données chargées. Forme: {df.shape}")

    # Définition du Pipeline
    pipeline = Pipeline([
        ('handle_infinity', InfinityHandler()),
        ('imputer', SimpleImputer(strategy='median')),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=15,           # <-- RÉDUIRE
            min_samples_leaf=10,    # <-- AUGMENTER (très important)
            class_weight=None, 
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    print("\nPipeline défini :")
    print(pipeline)

    # Séparation des données (pour évaluation locale)
    print("\nSéparation des données pour l'évaluation locale...")
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

    # Entraînement pour évaluation locale
    print("\nDébut de l'entraînement (évaluation locale)...")
    pipeline.fit(X_train, y_train)
    print("Entraînement local terminé.")

    # Évaluation locale
    print("\nÉvaluation sur le jeu de test local...")
    y_pred = pipeline.predict(X_test)
    local_score = accuracy_score(y_test, y_pred)

    print(f"Score (Accuracy) local : {local_score:.4f}")
    print("\nRapport de classification local :")
    print(classification_report(y_test, y_pred))

    # Entraînement final sur TOUTES les données équilibrées
    print("\nDébut de l'entraînement final (sur 100% des données 10k/classe)...")
    pipeline.fit(X, y)
    print("Entraînement final terminé.")

    # Sauvegarde du modèle
    print("\nSauvegarde du modèle final sous 'student_model.skio'...")
    with open("student_model.skio", "wb") as f:
        skio.dump(pipeline, f)

    print("Modèle sauvegardé avec succès.")
    print("Mission terminée. Vous pouvez soumettre 'student_model.skio'.")