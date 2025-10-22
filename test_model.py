#!/usr/bin/env python3
import argparse
import pandas as pd
import skops.io as skio

def main():
    # --- Définition des arguments ---
    parser = argparse.ArgumentParser(description="Évalue un modèle skio sur un fichier parquet.")
    parser.add_argument(
        "--data", 
        required=True, 
        help="Chemin vers le fichier .parquet (ex: Testing.parquet)"
    )
    parser.add_argument(
        "--model", 
        required=True, 
        help="Chemin vers le fichier modèle .skio (ex: student_model.skio)"
    )
    parser.add_argument(
        "--label", 
        default="ClassLabel", 
        help="Nom de la colonne cible (par défaut: ClassLabel)"
    )
    
    args = parser.parse_args()

    # --- Chargement des données ---
    df = pd.read_parquet(args.data)

    # --- Chargement du modèle ---
    with open(args.model, "rb") as f:
        from_model = skio.load(f)

    # --- Séparation X / y ---
    X = df.drop(columns=[args.label], axis=1)
    y = df[args.label]

    # --- Calcul du score ---
    score = from_model.score(X, y)
    print(f"Score du modèle : {score:.4f}")


if __name__ == "__main__":
    main()
