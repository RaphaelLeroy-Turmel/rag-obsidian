# Nettoyage : on sépare les metadonnées du corps de texte
# ignore les images
# wikilinks + embeded : [[note]], [[note|alias]] + ![[image.png]], ![[autre note]]  on supprimer pour la v1 du script a vori plus tard si on fait autrement
# pathlib , re, unicodedata, dataclasses

from dataclasses import dataclass, field
@dataclass
class cleanFile:
    # on supprimes les tags et autre propriété YAML, on ne garde que le titre, les liens vers autres notes
    name: str # name of the docment (title)
    content : str # le contenu de la note nettoyée
    metadata: dict = field(default_factory=dict)  # Dictionnaire des propriétés YAML
    #wikilinks: list[str] = field(default_factory=list) # list des lien aux autre notes

    def __str__(self):
        # On limite l'affichage du contenu aux 400 premiers caractères pour ne pas polluer le terminal
        apercu_contenu = self.content[:400] + "..." if len(self.content) > 400 else self.content
        
        return (
            f"Fichier : {self.name}\n"
            f"  Métadonnées : {self.metadata}\n"
            f"  Contenu (aperçu) : {apercu_contenu}\n"
            f"{'='*40}"
        )

import os
from pathlib import Path
def load_env_file(filepath= ".env"):
    env_file = Path(filepath)
    env_vars = {}
    if not env_file.exists():
        print(f"! ERROR : .env file not found with path : ${filepath}")
        return {}
    
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        # avoiding empty line or comments
        if not line or line.startswith("#"):
            continue

        # separating var name (key) ad its value (value)
        key,value = line.split("=",1)
        key = key.strip()
        value = value.strip().strip("'\"")

        env_vars[key] = value
        os.environ[key] = value
    return env_vars

import re
def clean_wikilinks(text):
    # La regex
    pattern = r'!?(?:\[\[)([^\]\|#]+)(?:#[^\]\|]+)?(?:\|[^\]]+)?(?:\]\])'
    
    # Remplacement : \1 fait référence au groupe capturant (le <link>)
    cleaned_text = re.sub(pattern, r'\1', text)
    
    return cleaned_text

import frontmatter
def clean_files():
    storage_path = Path(os.getenv("STORAGE_DIR"))
    clean_dir_path = Path(os.getenv("CLEAN_DIR"))
    clean_dir = {}
    for i,file in enumerate(storage_path.iterdir()):
        if(file.suffix == ".md"):
            print(f"cleaning of file : {file.name}")
            #clean_file = Path(f"{clean_dir_path}{file.name}_cleaned")

            name = file.name
            text = clean_wikilinks(file.read_text(encoding="utf-8"))
            post = frontmatter.loads(text)   
            metadata = post.metadata  # Dictionnaire des propriétés YAML
            content = post.content    # Le contenu Markdown sans le YAML   
            cleanFile_i = cleanFile(name, content, metadata)
            clean_dir[name] = cleanFile_i
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





