# chunking.py : cut cleanFile in chunks to be tokznized/embedded
import hashlib                       # pour fabriquer l'identifiant (hash) de chaque chunk
import re                            # expressions régulières (détection des titres, coupes)
from dataclasses import dataclass, field
import json
from dataclasses import asdict
from pathlib import Path

from transformers import AutoTokenizer   # sert uniquement à COMPTER les tokens

# --- Paramètres réglables ---------------------------------------------------

MODEL_NAME = "intfloat/multilingual-e5-small"   # same tokenizer as embedding model
MAX_TOKENS = 300   # maxsize of a chunk , must be under 512 token because of tokenizer limits
MIN_TOKENS = 50
# Markdown title 1 to 6 "#" 
# group(1) = "##...#" level of title, group(2) = text of title
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")

# Niveaux de découpage, du plus grossier au plus fin.
# Chaque entrée = (séparateur pour recoller les morceaux, regex pour couper).
#   1. paragraphes : coupe sur les lignes vides
#   2. lignes      : coupe sur les retours à la ligne
#   3. phrases     : coupe après . ! ? suivi d'un espace
SEPARATORS = [("\n\n", r"\n\s*\n"), ("\n", r"\n"), (" ", r"(?<=[.!?])\s+")]

# Le tokenizer est chargé une seule fois (au premier besoin), puis réutilisé.
_tokenizer = None

def get_tokenizer():
    """Charge le tokenizer au premier appel, puis renvoie toujours le même."""
    global _tokenizer   # sans "global", Python croirait que _tokenizer est une variable locale
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return _tokenizer


def count_tokens(text):
    """Compte les tokens d'un texte avec le tokenizer du modèle d'embedding."""
    return len(get_tokenizer().encode(text, add_special_tokens=False))


def split_by_tokens(text: str, max_tokens: int) -> list[str]:
    """Dernier recours : coupe le texte tous les max_tokens tokens.

    On utilise les offsets (position de chaque token dans le texte d'origine)
    pour couper le texte ORIGINAL, au lieu de faire decode() : le texte reste
    identique au caractère près (espaces, accents, retours à la ligne).
    """
    encoding = get_tokenizer()(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoding["offset_mapping"]   # liste de (début, fin) en caractères, un par token
    pieces = []
    for i in range(0, len(offsets), max_tokens):
        window = offsets[i:i + max_tokens]
        start = window[0][0]
        # Fin de la tranche = début du premier token de la tranche suivante,
        # pour ne perdre aucun caractère entre deux tranches
        end = offsets[i + max_tokens][0] if i + max_tokens < len(offsets) else len(text)
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
    return pieces
@dataclass
class Chunk:
    """Un morceau de note : l'unité qui sera transformée en 1 vecteur."""
    text: str                                        # le texte du chunk (pour l'affichage et les citations)
    source: str                                      # nom du fichier d'origine
    headings: list[str] = field(default_factory=list)  # chemin de titres, ex. ["Historique", "Tests"]
    chunk_id: str = ""  

    @property
    def text_for_embedding(self):
        # text for the embedding model, E5 expect prefix "passage" for content and "query" for questions
        path = " > ".join(self.headings)
        body = f"{path}\n{self.text}" if path else self.text
        return f"passage: {body}"
    
    def __str__(self):
        #debug print : header, nb of tokens and preveiw 80 char
        preview = self.text[:80].replace("\n", " ")
        return f"[{' > '.join(self.headings)}] ({count_tokens(self.text)} tokens) \npreview:\n{preview}..."


def split_sections(content: str) -> list[tuple[list[str], str]]:
    """Découpe par titres Markdown, en ignorant les # dans les blocs de code.

    Retourne une liste de (chemin de titres, texte de la section).
    """
    sections = []
    headings = []     # pile de (niveau, titre) : le chemin de titres courant
    buffer = []       # lignes de la section en cours de lecture
    in_code = False   # True quand on est à l'intérieur d'un bloc ``` ... ```

    def flush():
        # Termine la section en cours : on enregistre (titres, texte) si le texte n'est pas vide
        text = "\n".join(buffer).strip()
        if text:
            sections.append(([title for _, title in headings], text))
        buffer.clear()   # on repart avec un buffer vide pour la section suivante

    for line in content.splitlines():
        # Une ligne qui commence par ``` ouvre ou ferme un bloc de code
        if line.strip().startswith("```"):
            in_code = not in_code
        # Dans un bloc de code, un "#" n'est pas un titre (ex. un commentaire Python)
        match = None if in_code else HEADING_RE.match(line)
        if match:
            flush()                       # un nouveau titre ferme la section précédente
            level = len(match.group(1))   # nombre de "#" = niveau du titre
            # On retire de la pile les titres de niveau égal ou inférieur :
            # un nouveau "##" remplace l'ancien "##" et ses sous-titres
            while headings and headings[-1][0] >= level:
                headings.pop()
            headings.append((level, match.group(2)))
        else:
            buffer.append(line)           # ligne normale : elle appartient à la section en cours
    flush()   # ne pas oublier la dernière section à la fin du fichier
    return sections


def pack(units: list[str], sep: str, max_tokens: int) -> list[str]:
    """Regroupe des morceaux consécutifs tant que ça tient dans max_tokens."""
    pieces = []
    current = ""   # le groupe en cours de construction
    for unit in units:
        # On essaie d'ajouter le morceau au groupe courant
        candidate = f"{current}{sep}{unit}" if current else unit
        # Si le groupe est vide, on accepte toujours (même si le morceau seul est trop gros)
        if not current or count_tokens(candidate) <= max_tokens:
            current = candidate
        else:
            # Ça dépasse : on ferme le groupe courant et on en démarre un nouveau
            pieces.append(current)
            current = unit
    if current:
        pieces.append(current)   # dernier groupe
    return pieces


def split_text(text: str, max_tokens: int, level: int = 0) -> list[str]:
    """Paragraphes d'abord, puis lignes, puis phrases (récursif)."""
    # Cas d'arrêt : le texte est assez petit
    if count_tokens(text) <= max_tokens:
        return [text]
    # Plus aucun séparateur ne fonctionne (phrase géante, code, PDF sans ponctuation) :
    # on coupe directement sur les tokens
    if level >= len(SEPARATORS):
        return split_by_tokens(text, max_tokens)
    sep, pattern = SEPARATORS[level]
    # On coupe au niveau courant et on retire les morceaux vides
    units = [u for u in re.split(pattern, text) if u.strip()]
    # Si ce niveau ne coupe rien (un seul morceau), on passe directement au niveau plus fin
    if len(units) <= 1:
        return split_text(text, max_tokens, level + 1)
    # Chaque morceau encore trop gros est redécoupé au niveau suivant
    flat = []
    for unit in units:
        flat.extend(split_text(unit, max_tokens, level + 1))
    # Puis on regroupe les petits morceaux pour remplir les chunks jusqu'à max_tokens
    return pack(flat, sep, max_tokens)

def merge_small(pieces: list[tuple[list[str], str]]) -> list[tuple[list[str], str]]:
    """Fusionne les chunks trop courts avec le chunk précédent."""
    merged = []
    for headings, text in pieces:
        # Chunk trop court ET il existe un chunk précédent : on tente la fusion
        if merged and count_tokens(text) < MIN_TOKENS:
            prev_headings, prev_text = merged[-1]
            # Si on change de section, on remet le titre au début du texte ajouté,
            # pour ne pas perdre l'information de quelle section il vient
            extra = text if headings == prev_headings or not headings else f"{headings[-1]}\n{text}"
            combined = f"{prev_text}\n\n{extra}"
            # On ne fusionne que si le résultat reste sous la limite
            if count_tokens(combined) <= MAX_TOKENS:
                merged[-1] = (prev_headings, combined)   # on remplace le précédent par la version fusionnée
                continue                                  # et on passe au chunk suivant
        merged.append((headings, text))   # sinon, on garde le chunk tel quel
    return merged

def make_id(source: str, headings: list[str], index: int, text: str) -> str:
    """Identifiant stable : même entrée = même id, texte modifié = id différent."""
    raw = f"{source}|{' > '.join(headings)}|{index}|{text}"
    # SHA-256 puis on garde les 16 premiers caractères (assez pour éviter les collisions ici)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

def chunk_file(cleaned) -> list[Chunk]:
    """Transforme un cleanFile en liste de Chunk."""
    pieces = []
    # Étape 1 : découpage par titres, puis étape 2 : contrôle de taille de chaque section
    for headings, text in split_sections(cleaned.content):
        for part in split_text(text, MAX_TOKENS):
            pieces.append((headings, part))
    # Étape 3 : fusion des morceaux trop petits
    pieces = merge_small(pieces)
    # Construction des objets Chunk, avec leur identifiant (i = position dans la note)
    return [
        Chunk(text, cleaned.name, headings, make_id(cleaned.name, headings, i, text))
        for i, (headings, text) in enumerate(pieces)
    ]


def chunk_files(cleaned_files: dict) -> list[Chunk]:
    """Applique chunk_file à toutes les notes nettoyées."""
    chunks = []
    for cleaned in cleaned_files.values():   # le dict est {nom: cleanFile}
        chunks.extend(chunk_file(cleaned))
    return chunks



def save_chunks(chunks: list[Chunk], out_file: Path) -> None:
    """Écrit les chunks en JSONL (une ligne = un chunk) pour pouvoir les relire."""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        for c in chunks:
            record = asdict(c)                       # text, source, headings, chunk_id
            record["tokens"] = count_tokens(c.text)  # pratique pour repérer les chunks trop gros ou trop petits
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

# Ce bloc ne s'exécute que si on lance `python chunking.py` directement
# (pas quand un autre fichier fait `import chunking`).
if __name__ == "__main__":
    from cleaning import clean_files, load_env_file,env_path
    load_env_file()                                   # charge le .env (chemins)
    all_chunks = chunk_files(clean_files())           # nettoyage puis chunking de tout le dossier
    sizes = sorted(count_tokens(c.text) for c in all_chunks)
    # Distribution des tailles : sert à vérifier qu'aucun chunk ne dépasse la limite
    print(f"{len(all_chunks)} chunks | min {sizes[0]} | médiane {sizes[len(sizes)//2]} | max {sizes[-1]}")
    for c in all_chunks[:10]:                         # aperçu des 10 premiers chunks
        print(c)
        
    save_chunks(all_chunks, env_path("CLEAN_DIR") / "chunks.jsonl")