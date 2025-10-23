import argparse
import pandas as pd
import sys

def load_parquet(path: str) -> pd.DataFrame:
    """Charge un fichier Parquet et retourne un DataFrame."""
    return pd.read_parquet(path)

def save_csv(df: pd.DataFrame, path: str) -> None:
    """Sauvegarde un DataFrame au format CSV."""
    df.to_csv(path, index=False)

def main(argv=None):
    parser = argparse.ArgumentParser(description="Exporter un fichier Parquet en CSV.")
    parser.add_argument("-i", "--input", default="data.parquet", help="Chemin du fichier Parquet source.")
    parser.add_argument("-o", "--output", default="data.csv", help="Chemin du fichier CSV de destination.")
    parser.add_argument("-n", "--num-rows", type=int, default=None,
                        help="Nombre de lignes à exporter (toutes si non précisé).")
    args = parser.parse_args(argv)

    if args.num_rows is not None and args.num_rows < 0:
        parser.error("Le nombre de lignes doit être positif ou non spécifié.")

    try:
        full_df = load_parquet(args.input)
    except Exception as e:
        print(f"Erreur lors du chargement du fichier Parquet: {e}", file=sys.stderr)
        sys.exit(1)

    loaded_rows = len(full_df)
    if args.num_rows is not None:
        df = full_df.head(args.num_rows)
    else:
        df = full_df
    exported_rows = len(df)

    print(f"Fichier Parquet chargé avec {loaded_rows} lignes et {len(full_df.columns)} colonnes.")
    save_csv(df, args.output)
    print(f"{exported_rows} lignes exportées vers : {args.output}")

if __name__ == "__main__":
    main()