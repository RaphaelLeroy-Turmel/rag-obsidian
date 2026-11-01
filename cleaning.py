# Nettoyage : on sépare les metadonnées du corps de texte
# ignore les images
# wikilinks + embeded : [[note]], [[note|alias]] + ![[image.png]], ![[autre note]]  on supprimer pour la v1 du script a vori plus tard si on fait autrement
# pathlib , re, unicodedata, dataclasses

from dataclasses import dataclass, field
import os
from pathlib import Path
import frontmatter
from dotenv import dotenv_values
import re

@dataclass
class cleanFile:
    # delete tags, YAML porpreties, keep only title and link to other notes
    name: str # name of the docment (title)
    content : str # le contenu de la note nettoyée
    metadata: dict = field(default_factory=dict)  # Dictionnaire des propriétés YAML
    #wikilinks: list[str] = field(default_factory=list) # list

    def __str__(self):
        # only print first 400 characters
        apercu_contenu = self.content[:400] + "..." if len(self.content) > 400 else self.content
        
        return (
            f"Fichier : {self.name}\n"
            f"  Métadonnées : {self.metadata}\n"
            f"  Contenu (aperçu) : {apercu_contenu}\n"
            f"{'='*40}"
        )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_env_file(filepath=".env") -> dict:
    """Load KEY=VALUE pairs from a .env file into os.environ and return them."""
    path = Path(filepath)
    if not path.exists():
        print(".env file not found: ", path)
        return {}
    # dotenv handles comments, quotes and "=" inside values;
    # malformed lines come back with a None value, so we skip them
    values = {k: v for k, v in dotenv_values(path).items() if v is not None}
    os.environ.update(values)
    return values


# ---------------------------------------------------------------------------
# Text cleaning rules 
# ---------------------------------------------------------------------------
# later I will add more text cleaning rules


def clean_wikilinks(text):
    # La regex
    pattern = r'!?(?:\[\[)([^\]\|#]+)(?:#[^\]\|]+)?(?:\|[^\]]+)?(?:\]\])'
    
    # Remplacement : \1 fait référence au groupe capturant (le <link>)
    cleaned_text = re.sub(pattern, r'\1', text)
    
    return cleaned_text

def clean_files():
    storage_path = Path(os.getenv("STORAGE_DIR"))
    clean_dir_path = Path(os.getenv("CLEAN_DIR"))
    clean_dir = {}
    for i,file in enumerate(storage_path.iterdir()):
        if(file.suffix == ".md"):
            print(f"cleaning of file : {file.name}")
            

            name = file.name
            text = clean_wikilinks(file.read_text(encoding="utf-8"))
            post = frontmatter.loads(text)   
            metadata = post.metadata  # Dictionnaire des propriétés YAML
            content = post.content    # Le contenu Markdown sans le YAML   
            cleanFile_i = cleanFile(name, content, metadata)
            clean_dir[name] = cleanFile_i
            clean_file = Path(f"{clean_dir_path}{file.name}_cleaned")
            clean_file.touch()
            clean_file.write_text(content)
            clean_file.unlink()

    return clean_dir




def main():
    # charger .env
    config = load_env_file()
    for k,v in config.items():
        print(f"{k}={v}")
    
    clean_dir = clean_files()
    for k,v in clean_dir.items():
        print(f"item name : {k}\nvalue :\n{v}")



if __name__== "__main__":
    main()





