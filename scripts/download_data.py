"""Téléchargement du vrai dataset depuis une *GitHub Release* (hors historique git).

POURQUOI ce script (compétence C3.1 — reproductibilité) ?
---------------------------------------------------------
Le vrai `male_players.csv` (~90 Mo) ne doit PAS vivre dans l'historique git :
GitHub bloque les fichiers > 100 Mo, avertit dès 50 Mo, et un gros binaire alourdit
définitivement les clones. La pratique professionnelle est de publier le fichier comme
**asset d'une GitHub Release** (jusqu'à 2 Go, hors dépôt) et de le récupérer à la demande.

Ce script télécharge l'asset dans `data/raw/`. Il gère un asset `.csv` ou `.csv.gz`
(décompression automatique), respecte le proxy (variables d'environnement), et ne
re-télécharge pas si le fichier est déjà présent (sauf `--force`).

URL de l'asset, par ordre de priorité :
  1. argument `--url`
  2. variable d'environnement `DATA_URL`
  3. URL construite depuis `--repo` et `--tag` (défaut ci-dessous)

Exemples :
    python scripts/download_data.py
    python scripts/download_data.py --tag data-v1 --asset male_players.csv.gz
    DATA_URL=https://.../male_players.csv python scripts/download_data.py
"""
from __future__ import annotations

import argparse
import gzip
import os
import shutil
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import load_config  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

# Valeurs par défaut — à adapter à TA release (cf. README, section « Données réelles »).
DEFAULT_REPO = "BoucherAnthony0/App"
DEFAULT_TAG = "data-v1"
DEFAULT_ASSET = "male_players.csv"


def _build_url(repo: str, tag: str, asset: str) -> str:
    return f"https://github.com/{repo}/releases/download/{tag}/{asset}"


def _download_with_resume(url: str, tmp: str, max_retries: int = 5) -> None:
    """Télécharge avec reprise (Range) et retries — robuste aux coupures de proxy.

    POURQUOI : un téléchargement de ~90 Mo via le proxy peut se couper en fin de transfert.
    On reprend alors là où on s'est arrêté (en-tête HTTP `Range`) plutôt que tout reprendre.
    """
    total = None
    for attempt in range(1, max_retries + 1):
        done = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={done}-"} if done else {})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                # Taille totale (Content-Range si reprise, sinon Content-Length).
                if total is None:
                    cr = resp.headers.get("Content-Range")
                    total = int(cr.split("/")[-1]) if cr else (
                        int(resp.headers.get("Content-Length", 0)) + done
                    )
                mode = "ab" if done else "wb"
                with open(tmp, mode) as f:
                    while True:
                        chunk = resp.read(1024 * 256)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        if total:
                            sys.stdout.write(
                                f"\r  {done/1e6:6.1f}/{total/1e6:.1f} Mo "
                                f"({done*100//total:3d} %)"
                            )
                            sys.stdout.flush()
            if total and os.path.getsize(tmp) >= total:
                sys.stdout.write("\n")
                return
        except Exception as exc:
            if attempt == max_retries:
                raise
            logger.warning("Coupure (%s) — reprise %d/%d…", exc, attempt + 1, max_retries)
    sys.stdout.write("\n")


def download(url: str, dest: str, force: bool = False) -> str:
    """Télécharge `url` vers `dest` (décompresse si l'URL finit par .gz)."""
    if os.path.exists(dest) and not force:
        logger.info("Déjà présent : %s (utiliser --force pour re-télécharger)", dest)
        return dest

    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    is_gz = url.endswith(".gz")
    tmp = dest + (".gz.part" if is_gz else ".part")

    logger.info("Téléchargement depuis %s", url)
    try:
        _download_with_resume(url, tmp)
    except Exception as exc:  # 404 = release/asset inexistant, réseau épuisé, etc.
        if os.path.exists(tmp):
            os.remove(tmp)
        raise SystemExit(
            f"\nÉchec du téléchargement ({exc}).\n"
            f"Vérifie que l'asset existe bien sur la Release "
            f"(URL : {url}). Voir README section « Données réelles »."
        )

    if is_gz:
        logger.info("Décompression .gz → %s", dest)
        with gzip.open(tmp, "rb") as f_in, open(dest, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(tmp)
    else:
        os.replace(tmp, dest)

    size_mb = os.path.getsize(dest) / 1e6
    logger.info("OK → %s (%.1f Mo)", dest, size_mb)
    return dest


def main() -> None:
    config = load_config("config/params.yaml")
    default_dest = config["data"]["raw_path"]

    parser = argparse.ArgumentParser(description="Télécharge le dataset depuis une GitHub Release.")
    parser.add_argument("--url", help="URL directe de l'asset (prioritaire).")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="owner/repo de la Release.")
    parser.add_argument("--tag", default=DEFAULT_TAG, help="Tag de la Release (ex. data-v1).")
    parser.add_argument("--asset", default=DEFAULT_ASSET, help="Nom du fichier asset.")
    parser.add_argument("--out", default=default_dest, help="Chemin de destination.")
    parser.add_argument("--force", action="store_true", help="Re-télécharge même si présent.")
    args = parser.parse_args()

    url = args.url or os.environ.get("DATA_URL") or _build_url(args.repo, args.tag, args.asset)
    download(url, args.out, force=args.force)


if __name__ == "__main__":
    main()
