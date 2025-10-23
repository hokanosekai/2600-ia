#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel

# Imbalanced-learn (resampling)
try:
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTEENN, SMOTETomek
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAVE_IMBLEARN = True
except Exception:
    HAVE_IMBLEARN = False

# SKOPS pour la sérialisation
try:
    from skops import io as skio
    HAVE_SKOPS = True
except Exception:
    HAVE_SKOPS = False
    import joblib

# XGBoost et LightGBM (optionnels)
try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAVE_LGB = True
except Exception:
    HAVE_LGB = False


def load_any(path: Path) -> pd.DataFrame:
    """Charge les données depuis un fichier Parquet ou CSV."""
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        try:
            return pd.read_parquet(path)
        except Exception as e:
            print("⚠️ Impossible de lire Parquet sans pyarrow/fastparquet:", e, file=sys.stderr)
            raise
    elif path.suffix.lower() in {".csv", ".txt"}:
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.read_csv(path, sep=";")
    else:
        raise ValueError(f"Format non supporté: {path.suffix}")


def find_target(df: pd.DataFrame) -> str:
    """Trouve la colonne cible probable dans le DataFrame."""
    for c in ["ClassLabel", "label", "Label", "class", "target", "y"]:
        if c in df.columns:
            return c
    # Fallback: colonne de type objet avec peu de valeurs uniques
    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    for c in reversed(obj_cols):
        if df[c].nunique(dropna=True) <= max(20, int(0.05 * len(df))):
            return c
    raise RuntimeError("Colonne cible introuvable. Ajoutez --target <nom_colonne>.")


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crée des features dérivées pour améliorer la performance."""
    df = df.copy()
    
    # Ratios et features dérivées
    if 'Total Fwd Packets' in df.columns and 'Total Backward Packets' in df.columns:
        df['Total_Packets'] = df['Total Fwd Packets'] + df['Total Backward Packets']
        df['Fwd_Bwd_Ratio'] = df['Total Fwd Packets'] / (df['Total Backward Packets'] + 1)
        df['Fwd_Packet_Ratio'] = df['Total Fwd Packets'] / (df['Total_Packets'] + 1)
    
    if 'Fwd Packets Length Total' in df.columns and 'Bwd Packets Length Total' in df.columns:
        df['Total_Bytes'] = df['Fwd Packets Length Total'] + df['Bwd Packets Length Total']
        df['Bytes_per_Packet'] = df['Total_Bytes'] / (df['Total_Packets'] + 1)
        df['Fwd_Bwd_Bytes_Ratio'] = df['Fwd Packets Length Total'] / (df['Bwd Packets Length Total'] + 1)
    
    # Features de temporalité
    if 'Flow Duration' in df.columns:
        df['Bytes_per_Second'] = df['Total_Bytes'] / (df['Flow Duration'] + 1)
        df['Packets_per_Second'] = df['Total_Packets'] / (df['Flow Duration'] + 1)
        df['Flow_Duration_Log'] = np.log1p(df['Flow Duration'])
    
    # Features statistiques avancées
    if 'Fwd Packet Length Mean' in df.columns and 'Bwd Packet Length Mean' in df.columns:
        df['Avg_Packet_Size_Diff'] = df['Fwd Packet Length Mean'] - df['Bwd Packet Length Mean']
        df['Avg_Packet_Size_Sum'] = df['Fwd Packet Length Mean'] + df['Bwd Packet Length Mean']
    
    # Features de flags
    flag_columns = [col for col in df.columns if 'Flag' in col or 'URG' in col or 'SYN' in col]
    if flag_columns:
        df['Total_Flags'] = df[flag_columns].sum(axis=1)
    
    return df


def enhanced_build_preprocess(X: pd.DataFrame) -> ColumnTransformer:
    """Construit le pipeline de prétraitement amélioré."""
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("num", numeric_pipeline, num_cols),
        ("cat", categorical_pipeline, cat_cols),
    ], remainder="passthrough", sparse_threshold=0)


def enhanced_resampling_strategy(X, y, method='smote_tomek', sampling_strategy='auto'):
    """Stratégie de rééchantillonnage adaptative."""
    if not HAVE_IMBLEARN:
        return None
        
    class_counts = y.value_counts()
    min_class = class_counts.min()
    
    # Ajuster k_neighbors pour les petites classes
    k_neighbors = min(5, min_class - 1) if min_class > 1 else 1
    
    if method == 'smote':
        return SMOTE(
            random_state=42, 
            sampling_strategy=sampling_strategy, 
            k_neighbors=k_neighbors
        )
    elif method == 'smote_tomek':
        return SMOTETomek(
            random_state=42, 
            sampling_strategy=sampling_strategy,
            smote=SMOTE(k_neighbors=k_neighbors)
        )
    elif method == 'smote_enn':
        return SMOTEENN(
            random_state=42, 
            sampling_strategy=sampling_strategy,
            smote=SMOTE(k_neighbors=k_neighbors)
        )
    elif method == 'adasyn':
        return ADASYN(random_state=42, sampling_strategy=sampling_strategy)
    elif method == 'under':
        return RandomUnderSampler(random_state=42, sampling_strategy=sampling_strategy)
    
    return None


def make_enhanced_model(kind: str, random_state: int) -> object:
    """Crée une instance de modèle de classification amélioré."""
    kind = kind.lower()
    
    if kind == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=1000,
            learning_rate=0.05,
            max_depth=12,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=random_state,
            class_weight='balanced'
        )
    elif kind == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            class_weight='balanced',
            n_jobs=-1,
            random_state=random_state
        )
    elif kind == "logreg":
        return LogisticRegression(
            max_iter=2000,
            class_weight='balanced',
            C=0.1,
            solver='liblinear',
            random_state=random_state
        )
    elif kind == "xgb" and HAVE_XGB:
        return XGBClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            eval_metric='mlogloss',
            use_label_encoder=False
        )
    elif kind == "lgb" and HAVE_LGB:
        return LGBMClassifier(
            n_estimators=500,
            max_depth=10,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            class_weight='balanced'
        )
    else:
        # Fallback vers HGB si modèle demandé non disponible
        if kind in ["xgb", "lgb"]:
            print(f"⚠️ {kind.upper()} non disponible, utilisation de HGB à la place.")
        return HistGradientBoostingClassifier(
            max_iter=500,
            random_state=random_state,
            class_weight='balanced'
        )


def optimize_hyperparameters(X, y, model_type='hgb', cv=3, n_iter=10):
    """Optimisation des hyperparamètres avec RandomizedSearchCV."""
    preprocessor = enhanced_build_preprocess(X)
    
    if model_type == 'hgb':
        model = HistGradientBoostingClassifier(random_state=42, class_weight='balanced')
        param_dist = {
            'model__max_iter': [500, 1000, 1500],
            'model__learning_rate': [0.01, 0.05, 0.1],
            'model__max_depth': [8, 12, 15],
            'model__min_samples_leaf': [10, 20, 30],
            'model__l2_regularization': [0.1, 1.0, 10.0]
        }
    elif model_type == 'rf':
        model = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
        param_dist = {
            'model__n_estimators': [200, 300, 500],
            'model__max_depth': [10, 15, 20, None],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4],
            'model__max_features': ['sqrt', 'log2', None]
        }
    elif model_type == 'xgb' and HAVE_XGB:
        model = XGBClassifier(random_state=42, use_label_encoder=False)
        param_dist = {
            'model__n_estimators': [200, 500, 800],
            'model__max_depth': [6, 8, 10],
            'model__learning_rate': [0.01, 0.1, 0.2],
            'model__subsample': [0.8, 0.9, 1.0],
            'model__colsample_bytree': [0.8, 0.9, 1.0]
        }
    else:
        print(f"Optimisation non disponible pour {model_type}, utilisation des paramètres par défaut.")
        return None

    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('model', model)
    ])

    search = RandomizedSearchCV(
        pipeline, param_dist, n_iter=n_iter, 
        cv=min(cv, 3), scoring='balanced_accuracy', 
        n_jobs=-1, random_state=42, verbose=1
    )
    
    print(f"🔍 Optimisation des hyperparamètres pour {model_type}...")
    search.fit(X, y)
    
    print(f"✅ Meilleurs paramètres: {search.best_params_}")
    print(f"✅ Meilleur score: {search.best_score_:.4f}")
    
    return search.best_estimator_


def analyze_errors(model, X_val, y_val, feature_names=None):
    """Analyse détaillée des erreurs de classification."""
    pred = model.predict(X_val)
    
    # Matrice de confusion
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_val, pred)
    
    # Analyse par classe
    misclassified = X_val[pred != y_val].copy()
    misclassified['True_Label'] = y_val[pred != y_val].values
    misclassified['Predicted_Label'] = pred[pred != y_val]
    
    error_analysis = misclassified.groupby('True_Label').size()
    print("\n--- Analyse des Erreurs ---")
    print("Nombre d'erreurs par classe réelle:")
    print(error_analysis)
    
    return misclassified


def main():
    """Fonction principale améliorée pour l'entraînement et l'export du modèle."""
    p = argparse.ArgumentParser(description="Script d'entraînement de modèle de classification amélioré.")
    p.add_argument("--train-file", default="Training.parquet", help="Fichier de données d'entraînement (Parquet ou CSV).")
    p.add_argument("--target", default=None, help="Nom de la colonne cible (détectée automatiquement si non fournie).")
    p.add_argument("--model", default="hgb", choices=["hgb", "rf", "logreg", "xgb", "lgb"], help="Algorithme principal à utiliser.")
    p.add_argument("--cv", type=int, default=5, help="Nombre de folds pour la validation croisée (0 pour désactiver).")
    p.add_argument("--subsample", type=int, default=0, help="Échantillonner N lignes pour un entraînement rapide (0 pour tout utiliser).")
    p.add_argument("--testsample", type=float, default=0.2, help="Fraction des données à utiliser comme jeu de test hold-out (0 pour désactiver).")
    p.add_argument("--seed", type=int, default=42, help="Graine aléatoire pour la reproductibilité.")
    p.add_argument(
        "--scoring",
        default="balanced_accuracy",
        choices=["accuracy", "f1_macro", "balanced_accuracy", "f1_weighted", "roc_auc_ovr"],
        help="Métrique utilisée pour la validation croisée.",
    )
    # Resampling options
    p.add_argument(
        "--resampling",
        choices=["none", "smote", "smote_tomek", "smote_enn", "under", "adasyn"],
        default="smote_tomek",
        help="Technique de ré-échantillonnage à utiliser pendant l'entraînement.",
    )
    p.add_argument(
        "--sampling-strategy",
        default="auto",
        help="Stratégie de sampling imbalanced-learn (ex: 'auto', 0.5, dict).",
    )
    p.add_argument(
        "--smote-k-neighbors",
        type=int,
        default=5,
        help="k_neighbors pour SMOTE (utilisé pour smote/smote_tomek/smote_enn).",
    )
    # Nouvelles options
    p.add_argument("--optimize", action="store_true", help="Activer l'optimisation des hyperparamètres.")
    p.add_argument("--feature-engineering", action="store_true", default=True, help="Activer le feature engineering.")
    p.add_argument("--analyze-errors", action="store_true", help="Analyser les erreurs de classification.")
    p.add_argument("--n-iter", type=int, default=15, help="Nombre d'itérations pour l'optimisation.")

    args = p.parse_args()

    print("=== ENTRAÎNEMENT AMÉLIORÉ DE MODÈLE ===")
    print("--- Étape 1: Chargement et Nettoyage des Données ---")
    df = load_any(Path(args.train_file))
    df.columns = df.columns.str.strip()
    
    # Remplacer les valeurs infinies par NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    print(f"Données chargées: {len(df)} lignes, {len(df.columns)} colonnes.")

    print("\n--- Étape 2: Préparation des Données ---")
    target_col = args.target or find_target(df)
    y = df[target_col].astype("category")
    X = df.drop(columns=[target_col])
    print(f"Colonne cible '{target_col}' identifiée. Distribution:")
    print(y.value_counts())

    # Feature engineering
    if args.feature_engineering:
        print("\n--- Feature Engineering ---")
        X = create_features(X)
        print(f"✅ Features après engineering: {X.shape[1]} colonnes")

    if args.subsample and args.subsample < len(df):
        indices = np.random.RandomState(args.seed).choice(len(df), args.subsample, replace=False)
        X = X.iloc[indices].copy()
        y = y.iloc[indices].copy()
        print(f"➡️ Sous-échantillon de {len(X)} lignes utilisé pour l'entraînement.")

    print("\n--- Étape 3: Création du Pipeline ---")
    preprocess_pipeline = enhanced_build_preprocess(X)
    model = make_enhanced_model(args.model, args.seed)

    def make_training_pipeline():
        """Construit la pipeline d'entraînement, avec resampling si demandé."""
        if args.resampling == "none":
            return Pipeline([("preprocess", preprocess_pipeline), ("model", model)])

        if not HAVE_IMBLEARN:
            raise RuntimeError("imbalanced-learn n'est pas installé mais --resampling est utilisé.")

        sampler = enhanced_resampling_strategy(
            X, y, 
            method=args.resampling,
            sampling_strategy=args.sampling_strategy
        )

        if sampler is None:
            return Pipeline([("preprocess", preprocess_pipeline), ("model", model)])
        
        return ImbPipeline([
            ("preprocess", preprocess_pipeline), 
            ("sampler", sampler), 
            ("model", model)
        ])

    # Optimisation des hyperparamètres
    if args.optimize:
        print("\n--- Optimisation des Hyperparamètres ---")
        best_pipe = optimize_hyperparameters(
            X, y, 
            model_type=args.model, 
            cv=args.cv, 
            n_iter=args.n_iter
        )
        if best_pipe is not None:
            pipe = best_pipe
        else:
            pipe = make_training_pipeline()
    else:
        pipe = make_training_pipeline()

    print("Pipeline créé avec succès:")
    print(pipe)

    # Validation croisée
    if args.cv > 0 and y.nunique() > 1:
        print(f"\n--- Étape 4a: Validation Croisée ({args.cv} folds) ---")
        splits = max(2, min(args.cv, int(y.value_counts().min())))
        if splits < args.cv:
            warnings.warn(f"CV réduite à {splits} folds en raison de classes peu représentées.")
        
        skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=args.seed)
        cv_scores = cross_val_score(pipe, X, y, cv=skf, scoring=args.scoring, n_jobs=-1)
        print(f"CV {args.scoring}: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"Détail des folds: {cv_scores}")

    # Évaluation sur un jeu de test (hold-out)
    test_accuracy = 0.0
    if y.nunique() > 1 and args.testsample > 0:
        print(f"\n--- Étape 4b: Évaluation sur Jeu de Test ({int(args.testsample*100)}% hold-out) ---")
        X_tr, X_va, y_tr, y_va = train_test_split(
            X, y, 
            test_size=args.testsample, 
            stratify=y, 
            random_state=args.seed
        )
        
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_va)
        
        # Métriques multiples
        test_accuracy = accuracy_score(y_va, pred)
        bal_accuracy = balanced_accuracy_score(y_va, pred)
        f1_macro = f1_score(y_va, pred, average='macro')
        
        print(f"Hold-out Accuracy: {test_accuracy:.4f}")
        print(f"Balanced Accuracy: {bal_accuracy:.4f}")
        print(f"F1 Macro: {f1_macro:.4f}")
        
        print("\nRapport de classification détaillé:")
        print(classification_report(y_va, pred, digits=4))
        
        # Analyse des erreurs
        if args.analyze_errors:
            misclassified = analyze_errors(pipe, X_va, y_va)
    else:
        # Pas de validation possible si une seule classe
        print("\n--- Entraînement sur toutes les données ---")
        pipe.fit(X, y)

    print("\n--- Étape 5: Entraînement Final et Export ---")
    # Ré-entraînement sur l'ensemble des données pour le modèle final
    pipe.fit(X, y)
    print("✅ Modèle final entraîné sur l'ensemble des données.")

    # Pour l'export, on retire la step de resampling pour éviter toute dépendance
    export_pipe = pipe
    if 'ImbPipeline' in globals() and isinstance(pipe, ImbPipeline) and "sampler" in dict(pipe.named_steps):
        export_pipe = Pipeline([
            ("preprocess", pipe.named_steps["preprocess"]),
            ("model", pipe.named_steps["model"]),
        ])
        print("✅ Pipeline d'export créé (sans resampling).")

    # Export du modèle
    timestamp = datetime.now().strftime('%d-%m-%Y')
    if args.testsample > 0:
        out_file_name = f"student_model.skio"
        # Sauvegarde avec le score aussi
        backup_name = f"{timestamp}-{args.model}-{args.resampling}-{test_accuracy:.4f}.skio"
    else:
        out_file_name = f"student_model.skio"
        backup_name = f"{timestamp}-{args.model}-{args.resampling}-final.skio"

    # Exporter le modèle principal
    if HAVE_SKOPS:
        with open(out_file_name, "wb") as f:
            skio.dump(export_pipe, f)
        print(f"✅ Modèle principal exporté: {out_file_name}")
        
        # Sauvegarde de backup
        with open(backup_name, "wb") as f:
            skio.dump(export_pipe, f)
        print(f"✅ Backup exporté: {backup_name}")
    else:
        alt_path = Path(out_file_name).with_suffix(".pkl")
        joblib.dump(export_pipe, alt_path)
        print(f"⚠️ skops non disponible. Modèle sauvegardé avec joblib: {alt_path}")

    print("\n=== ENTRAÎNEMENT TERMINÉ ===")


if __name__ == "__main__":
    with warnings.catch_warnings():
        # Ignorer certains avertissements pour une sortie plus propre
        warnings.filterwarnings('ignore', category=UserWarning, module='skops')
        warnings.filterwarnings('ignore', category=FutureWarning)
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        main()