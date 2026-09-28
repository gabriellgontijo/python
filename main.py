"""
Sistema de Recomendação Musical com FastAPI
===========================================

Atividade Prática: 5 APIs de recomendação sobre o dataset
"top50MusicFrom2010-2019.csv" (Top 50 do Spotify de 2010 a 2019).

    1. GET  /recommendations/content-based/{song_title}  -> similaridade por características
    2. POST /recommendations/genre-artist                -> mesmo gênero ou artista
    3. GET  /recommendations/collaborative/{user_id}     -> filtro colaborativo (co-ocorrências)
    4. POST /recommendations/hybrid                      -> conteúdo + colaborativo
    5. GET  /recommendations/popular                     -> mais populares por ano/gênero

Como executar localmente:
    pip install -r requirements.txt
    uvicorn main:app --reload
    Documentação interativa (Swagger): http://127.0.0.1:8000/docs
"""

import difflib
import json
import math
import os
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

app = FastAPI(
    title="Sistema de Recomendação Musical",
    description=(
        "Atividade Prática: 5 APIs de recomendação musical (conteúdo, gênero/artista, "
        "filtro colaborativo, híbrida e popularidade) usando o dataset Top 50 do Spotify "
        "de 2010 a 2019.\n\n"
        "Usuários fictícios do filtro colaborativo: `user1` a `user15` (veja `GET /users`)."
    ),
    version="1.0.0",
)


# =============================================================================
# Carregar dados
# =============================================================================

# O CSV do professor usa nomes de coluna longos e descritivos
# (ex.: "Energy- The energy of a song - the higher the value, ...").
# Eles são renomeados para os nomes curtos usados no código base do enunciado.
COLUMN_PREFIXES = {
    "title": "title",
    "artist": "artist",
    "the genre of the track": "genre",
    "year": "year",
    "beats.per.minute": "Beats.Per.Minute",
    "energy": "Energy",
    "danceability": "Danceability",
    "loudness/db": "Loudness/dB",
    "liveness": "Liveness",
    "valence": "Valence",
    "length": "Length",
    "acousticness": "Acousticness",
    "speechiness": "Speechiness",
    "popularity": "Popularity",
}

features = ['Beats.Per.Minute', 'Energy', 'Danceability', 'Loudness/dB',
            'Liveness', 'Valence', 'Length', 'Acousticness', 'Speechiness', 'Popularity']

# Caracteres chineses indicam texto corrompido ("mojibake"): o arquivo original em
# Windows-1252 foi lido como GBK antes de ser salvo em UTF-8 (ex.: "Se駉rita").
_MOJIBAKE = re.compile(r"[㐀-鿿豈-﫿]")
_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})


def find_csv() -> Path:
    """Procura o CSV na pasta deste arquivo ou na pasta atual (ex.: /content no Colab)."""
    if os.environ.get("MUSIC_CSV_PATH"):
        return Path(os.environ["MUSIC_CSV_PATH"])
    for folder in (Path(__file__).resolve().parent, Path.cwd()):
        found = sorted(folder.glob("top50*.csv"))
        if found:
            return found[0]
    raise FileNotFoundError(
        "Arquivo 'top50MusicFrom2010-2019.csv' não encontrado. "
        "Coloque o CSV na mesma pasta do main.py."
    )


def fix_mojibake(text: str) -> str:
    """Desfaz a corrupção de caracteres: 'Se駉rita' -> 'Señorita'."""
    if _MOJIBAKE.search(text):
        try:
            return text.encode("gbk").decode("cp1252")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return text
    return text


def normalize(text) -> str:
    """Chave de busca: minúsculas, sem acentos, aspas simples e espaços padronizados."""
    text = unicodedata.normalize("NFKD", str(text).translate(_QUOTES))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join(text.lower().split())


def load_dataset(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, encoding="utf-8")

    rename = {}
    for column in raw.columns:
        key = column.strip().lower()
        for prefix, name in COLUMN_PREFIXES.items():
            if key.startswith(prefix):
                rename[column] = name
                break
    data = raw.rename(columns=rename)
    required = ["title", "artist", "genre", "year"] + features
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Colunas não encontradas no CSV: {missing}")
    data = data[required].copy()

    data["original_title"] = data["title"].astype(str).str.strip()
    for column in ["title", "artist", "genre"]:
        data[column] = data[column].astype(str).str.strip().map(fix_mojibake)

    # Registro inválido: "Million Years Ago" (Adele) tem BPM 0 e todas as
    # características zeradas. Além de não ser uma música "real" para comparação,
    # ele distorceria a normalização (Loudness de -60 dB).
    data = data[data["Beats.Per.Minute"] > 0].reset_index(drop=True)

    # Chaves normalizadas usadas nas buscas (sem diferenciar maiúsculas/acentos)
    data["title_key"] = data["title"].map(normalize)
    data["original_title_key"] = data["original_title"].map(normalize)
    # Título "base", sem sufixos como "(feat. ...)" ou "- Remix": "Titanium" encontra "Titanium (feat. Sia)"
    data["base_title_key"] = data["title_key"].map(lambda t: re.split(r"\s+[-(\[]", t, maxsplit=1)[0])
    data["artist_key"] = data["artist"].map(normalize)
    data["genre_key"] = data["genre"].map(normalize)
    data["song_key"] = data["title_key"] + " | " + data["artist_key"]
    return data


df = load_dataset(find_csv())

# Pré-processamento: as características são normalizadas entre 0 e 1 (MinMaxScaler)
# em uma matriz separada, para que as respostas mostrem os valores originais.
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(df[features])

# Modelo de similaridade (cosseno entre todas as músicas)
similarity_matrix = cosine_similarity(features_scaled)

# 16 músicas aparecem duas vezes no CSV (ex.: "Love Yourself" em 2015 e 2016).
# Nas recomendações cada música aparece uma única vez (a versão mais popular).
songs = df.sort_values("Popularity", ascending=False, kind="mergesort").drop_duplicates("song_key")
canonical_index = dict(zip(songs["song_key"], songs.index))


# =============================================================================
# Dados fictícios para o filtro colaborativo
# =============================================================================

# "Usuários que gostaram disso também gostaram daquilo": cada usuário fictício tem
# uma lista de músicas curtidas. Os gostos formam grupos (pop dançante, pop/funk,
# EDM, pop canadense, baladas, latino) com algumas músicas em comum entre grupos.
user_likes: Dict[str, List[str]] = {
    # Pop dançante (Katy Perry, Lady Gaga, Rihanna, Kesha)
    "user1": ["Roar", "Dark Horse", "Bad Romance", "TiK ToK", "Only Girl (In The World)", "Telephone"],
    "user2": ["Roar", "Bad Romance", "Telephone", "Die Young", "Part Of Me", "Diamonds", "We Found Love"],
    "user3": ["Dark Horse", "TiK ToK", "Die Young", "Starships", "Problem", "Side To Side",
              "Titanium (feat. Sia)"],
    # Pop/funk (Bruno Mars, Maroon 5)
    "user4": ["Just the Way You Are", "Locked Out of Heaven", "Uptown Funk", "Sugar", "Payphone",
              "Treasure", "That's What I Like"],
    "user5": ["Uptown Funk", "That's What I Like", "24K Magic", "Sugar", "Animals", "Marry You",
              'Happy - From "Despicable Me 2"'],
    # EDM
    "user6": ["Wake Me Up", "Summer", "Titanium (feat. Sia)", "Clarity", "Don't Let Me Down", "Closer",
              "Hey Brother"],
    "user7": ["Wake Me Up", "Titanium (feat. Sia)", "This Is What You Came For", "Closer",
              "Something Just Like This", "Stay"],
    "user8": ["Summer", "How Deep Is Your Love", "One Kiss (with Dua Lipa)", "New Rules", "Silence",
              "Happier", "Don't Let Me Down"],
    # Pop canadense (Justin Bieber, Shawn Mendes, The Weeknd)
    "user9": ["Love Yourself", "Sorry", "What Do You Mean?", "Stitches", "Treat You Better", "Starboy",
              "The Hills"],
    "user10": ["Love Yourself", "Sorry", "Stitches", "Mercy", "There's Nothing Holdin' Me Back", "Baby",
               "Boyfriend"],
    "user11": ["Starboy", "The Hills", "I Feel It Coming", "Too Good", "Hold On, We're Going Home",
               "Love Yourself"],
    # Baladas / soul (Adele, Ed Sheeran, Sam Smith)
    "user12": ["Someone Like You", "Rolling in the Deep", "Set Fire to the Rain", "Thinking out Loud",
               "Stay With Me", "All of Me"],
    "user13": ["Someone Like You", "Thinking out Loud", "Stay With Me", "I'm Not The Only One", "All I Ask",
               "Let Her Go", "Shape of You"],
    # Latino / festa
    "user14": ["Despacito - Remix", "Havana (feat. Young Thug)", "Give Me Everything", "On The Floor",
               "Taki Taki (feat. Selena Gomez, Ozuna & Cardi B)", "International Love"],
    "user15": ["Despacito - Remix", "Havana (feat. Young Thug)", "Give Me Everything",
               "Hey Mama (feat. Nicki Minaj, Bebe Rexha & Afrojack)", "Shape of You", "Cheap Thrills"],
}


# =============================================================================
# Funções auxiliares
# =============================================================================

def match_title(title: str, artist: Optional[str] = None) -> pd.DataFrame:
    """Linhas do dataset cujo título corresponde à busca (exato ou título base)."""
    pool = df
    if artist and artist.strip():
        pool = pool[pool["artist_key"] == normalize(artist)]
    key = normalize(title)
    matches = pool[(pool["title_key"] == key) | (pool["original_title_key"] == key)]
    if matches.empty:
        matches = pool[pool["base_title_key"] == key]
    return matches


def suggest(query: str, options, n: int = 5) -> List[str]:
    """Sugestões de nomes parecidos, para mensagens de erro mais úteis."""
    by_key = {}
    for option in options:
        by_key.setdefault(normalize(option), option)
    key = normalize(query)
    found = difflib.get_close_matches(key, list(by_key), n=n, cutoff=0.5)
    if len(key) >= 3:
        found += [option_key for option_key in by_key if key in option_key]
    return [by_key[option_key] for option_key in list(dict.fromkeys(found))[:n]]


def find_song_index(song_title: str, artist: Optional[str] = None) -> int:
    """Índice da música buscada. Títulos repetidos: usa a versão mais popular."""
    matches = match_title(song_title, artist)
    if matches.empty:
        message = f"Música '{song_title}' não encontrada"
        message += f" para o artista '{artist}'." if artist else "."
        raise HTTPException(status_code=404, detail={
            "message": message,
            "suggestions": suggest(song_title, df["title"]),
        })
    return int(matches.sort_values("Popularity", ascending=False, kind="mergesort").index[0])


def resolve_user(user_id: str) -> str:
    """Aceita 'user1', 'USER1' ou apenas '1'."""
    key = normalize(user_id)
    if key.isdigit():
        key = f"user{int(key)}"
    for user in user_likes:
        if user.lower() == key:
            return user
    raise HTTPException(status_code=404, detail={
        "message": f"Usuário '{user_id}' não encontrado.",
        "available_users": list(user_likes),
    })


def song_info(row: pd.Series) -> dict:
    return {
        "title": row["title"],
        "artist": row["artist"],
        "genre": row["genre"],
        "year": int(row["year"]),
        "popularity": int(row["Popularity"]),
    }


def song_features(row: pd.Series) -> Dict[str, int]:
    return {feature: int(row[feature]) for feature in features}


def _feature_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", normalize(name))


FEATURE_NAMES = {_feature_key(feature): feature for feature in features}
FEATURE_NAMES.update({
    "bpm": "Beats.Per.Minute", "tempo": "Beats.Per.Minute",
    "loudness": "Loudness/dB", "db": "Loudness/dB", "duration": "Length",
})


def parse_weights(raw: Optional[str]) -> Optional[Dict[str, float]]:
    """Converte o parâmetro weights em um dicionário {característica: peso}.

    Aceita JSON ({"Energy": 2, "Danceability": 1.5}) ou o formato curto
    "Energy:2,Danceability:1.5". Características não informadas ficam com peso 1.
    """
    if raw is None or not raw.strip():
        return None
    text = raw.strip()
    invalid_format = HTTPException(status_code=400, detail={
        "message": 'Formato inválido para weights. Use JSON, ex.: {"Energy": 2, "Danceability": 1.5}, '
                   'ou "Energy:2,Danceability:1.5".',
        "valid_features": features,
    })
    try:
        if text.startswith("{"):
            parsed = json.loads(text)
            if not isinstance(parsed, dict):
                raise invalid_format
            items = list(parsed.items())
        else:
            items = []
            for part in text.split(","):
                name, separator, value = part.partition(":")
                if not separator:
                    name, separator, value = part.partition("=")
                if not separator:
                    raise invalid_format
                items.append((name, value))

        weights = {}
        for name, value in items:
            feature = FEATURE_NAMES.get(_feature_key(name))
            if feature is None:
                raise HTTPException(status_code=400, detail={
                    "message": f"Característica desconhecida em weights: '{name.strip()}'.",
                    "valid_features": features,
                })
            weight = float(value)
            if not math.isfinite(weight) or weight < 0:
                raise HTTPException(status_code=400, detail={
                    "message": f"O peso de '{feature}' deve ser um número maior ou igual a zero.",
                })
            weights[feature] = weight
    except (ValueError, TypeError):
        raise invalid_format

    full_weights = {feature: weights.get(feature, 1.0) for feature in features}
    if sum(full_weights.values()) == 0:
        raise HTTPException(status_code=400, detail={"message": "Pelo menos um peso deve ser maior que zero."})
    return full_weights


def weighted_similarity(index: int, weights: Dict[str, float]) -> np.ndarray:
    """Similaridade do cosseno ponderada: sum(w*x*y) / (sqrt(sum(w*x²)) * sqrt(sum(w*y²)))."""
    scale = np.sqrt([weights[feature] for feature in features])
    weighted = features_scaled * scale
    return cosine_similarity(weighted[index:index + 1], weighted)[0]


# Filtro colaborativo: cada música curtida pelos usuários fictícios é associada a
# uma linha do dataset e contamos quantas vezes cada par de músicas aparece junto.
user_song_keys: Dict[str, List[str]] = {}
for _user, _titles in user_likes.items():
    _keys = []
    for _title in _titles:
        _matches = match_title(_title)
        if _matches.empty:
            raise ValueError(f"Música de {_user} não encontrada no dataset: {_title}")
        _keys.append(df.at[int(_matches.sort_values("Popularity", ascending=False).index[0]), "song_key"])
    user_song_keys[_user] = list(dict.fromkeys(_keys))

# co_occurrence[A][B] = quantos usuários gostaram de A e também de B
co_occurrence: Dict[str, Counter] = defaultdict(Counter)
for _keys in user_song_keys.values():
    for _a in _keys:
        for _b in _keys:
            if _a != _b:
                co_occurrence[_a][_b] += 1


def collaborative_scores(user: str):
    """Score de cada música não curtida = soma das co-ocorrências com as músicas curtidas."""
    liked = set(user_song_keys[user])
    scores: Counter = Counter()
    reasons: Dict[str, List[str]] = defaultdict(list)
    for liked_key in user_song_keys[user]:
        for other_key, count in co_occurrence[liked_key].items():
            if other_key not in liked:
                scores[other_key] += count
                reasons[other_key].append(liked_key)
    return scores, reasons


def title_of(song_key: str) -> str:
    return df.at[canonical_index[song_key], "title"]


def short_info(song_key: str) -> dict:
    row = df.loc[canonical_index[song_key]]
    return {"title": row["title"], "artist": row["artist"]}


# =============================================================================
# Modelos de requisição (corpo dos POST)
# =============================================================================

class GenreArtistRequest(BaseModel):
    genre: Optional[str] = None
    artist: Optional[str] = None
    limit: int = Field(5, ge=1, le=100)

    model_config = ConfigDict(json_schema_extra={
        "examples": [{"genre": "canadian pop", "artist": "Ed Sheeran", "limit": 5}]
    })


class HybridRequest(BaseModel):
    song_title: str
    user_id: str
    content_weight: float = Field(0.7, ge=0)
    collab_weight: float = Field(0.3, ge=0)
    limit: int = Field(5, ge=1, le=100)

    model_config = ConfigDict(json_schema_extra={
        "examples": [{"song_title": "Shape of You", "user_id": "user12", "content_weight": 0.7,
                      "collab_weight": 0.3, "limit": 5}]
    })


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/", tags=["Auxiliares"], summary="Informações da API")
async def root():
    """Resumo do dataset e lista dos endpoints. A documentação interativa fica em `/docs`."""
    return {
        "message": "Sistema de Recomendação Musical",
        "docs": "/docs",
        "dataset": {
            "rows": len(df),
            "unique_songs": len(songs),
            "years": sorted(int(year) for year in df["year"].unique()),
            "genres": int(df["genre"].nunique()),
            "artists": int(df["artist"].nunique()),
        },
        "endpoints": [
            "GET  /recommendations/content-based/{song_title}",
            "POST /recommendations/genre-artist",
            "GET  /recommendations/collaborative/{user_id}",
            "POST /recommendations/hybrid",
            "GET  /recommendations/popular",
        ],
    }


@app.get("/users", tags=["Auxiliares"], summary="Usuários fictícios do filtro colaborativo")
async def list_users():
    """Lista os usuários simulados e as músicas que cada um curtiu."""
    return {
        "total": len(user_likes),
        "users": {user: [short_info(key) for key in keys] for user, keys in user_song_keys.items()},
    }


@app.get("/recommendations/content-based/{song_title:path}", tags=["Recomendações"],
         summary="1. Recomendação baseada em conteúdo")
async def content_based_recommendations(
    song_title: str,
    limit: int = Query(5, ge=1, le=100, description="Número de recomendações a retornar"),
    weights: Optional[str] = Query(
        None,
        description='Pesos por característica (dicionário em JSON). Ex.: {"Energy": 2, "Danceability": 1.5}. '
                    "Características não informadas ficam com peso 1; peso 0 ignora a característica.",
    ),
    artist: Optional[str] = Query(None, description="Opcional: artista, para títulos repetidos (ex.: 'Hello')"),
):
    """
    Retorna as músicas mais parecidas com `song_title` pelas características musicais
    (BPM, energia, danceabilidade, volume, ...), usando **similaridade do cosseno**
    sobre as características normalizadas (MinMaxScaler).

    - A busca do título não diferencia maiúsculas/minúsculas nem acentos, e aceita o
      título sem o complemento (ex.: `Titanium` encontra `Titanium (feat. Sia)`).
    - `weights` é um dicionário `{característica: peso}` enviado como JSON na query string.
      Características válidas: `Beats.Per.Minute`, `Energy`, `Danceability`, `Loudness/dB`,
      `Liveness`, `Valence`, `Length`, `Acousticness`, `Speechiness`, `Popularity`.

    **Exemplo de requisição**

        GET /recommendations/content-based/Shape%20of%20You?limit=2&weights={"Energy":2,"Danceability":2}

    **Exemplo de resposta** (`features` resumido)

        {
          "song": {"title": "Shape of You", "artist": "Ed Sheeran", "genre": "pop", "year": 2017,
                   "popularity": 87, "features": {"Beats.Per.Minute": 96, "Energy": 65, "...": "..."}},
          "weights": {"Beats.Per.Minute": 1.0, "Energy": 2.0, "Danceability": 2.0, "...": 1.0},
          "total": 2,
          "recommendations": [
            {"title": "Closer", "artist": "The Chainsmokers", "genre": "electropop", "year": 2017,
             "popularity": 86, "similarity": 0.9901, "features": {"Beats.Per.Minute": 95, "...": "..."}},
            {"title": "Sing", "artist": "Ed Sheeran", "genre": "pop", "year": 2015,
             "popularity": 71, "similarity": 0.9872, "features": {"Beats.Per.Minute": 120, "...": "..."}}
          ]
        }
    """
    index = find_song_index(song_title, artist)
    parsed_weights = parse_weights(weights)
    similarities = similarity_matrix[index] if parsed_weights is None else weighted_similarity(index, parsed_weights)

    seed = df.loc[index]
    candidates = songs[songs["song_key"] != seed["song_key"]].copy()
    candidates["similarity"] = similarities[candidates.index]
    top = candidates.sort_values(["similarity", "Popularity"], ascending=False, kind="mergesort").head(limit)

    return {
        "song": {**song_info(seed), "features": song_features(seed)},
        "weights": parsed_weights or {feature: 1.0 for feature in features},
        "total": len(top),
        "recommendations": [
            {**song_info(row), "similarity": round(float(row["similarity"]), 4), "features": song_features(row)}
            for _, row in top.iterrows()
        ],
    }


@app.post("/recommendations/genre-artist", tags=["Recomendações"],
          summary="2. Recomendação por gênero/artista")
async def genre_artist_recommendations(request: GenreArtistRequest):
    """
    Retorna músicas do **mesmo gênero OU do mesmo artista** informados, ordenadas por
    popularidade (maior primeiro). Informe `genre`, `artist` ou os dois. A comparação não
    diferencia maiúsculas/minúsculas nem acentos. O campo `match` indica o motivo da
    recomendação: `genre`, `artist` ou `genre+artist`.

    **Exemplo de requisição**

        POST /recommendations/genre-artist
        {"genre": "canadian pop", "artist": "Ed Sheeran", "limit": 3}

    **Exemplo de resposta**

        {
          "genre": "canadian pop",
          "artist": "Ed Sheeran",
          "total_matches": 42,
          "total": 3,
          "recommendations": [
            {"title": "Señorita", "artist": "Shawn Mendes", "genre": "canadian pop", "year": 2019,
             "popularity": 95, "match": "genre"},
            {"title": "South of the Border (feat. Camila Cabello & Cardi B)", "artist": "Ed Sheeran",
             "genre": "pop", "year": 2019, "popularity": 92, "match": "artist"},
            {"title": "Shape of You", "artist": "Ed Sheeran", "genre": "pop", "year": 2017,
             "popularity": 87, "match": "artist"}
          ]
        }
    """
    genre_key = normalize(request.genre) if request.genre and request.genre.strip() else None
    artist_key = normalize(request.artist) if request.artist and request.artist.strip() else None
    if genre_key is None and artist_key is None:
        raise HTTPException(status_code=400, detail={"message": "Informe 'genre' e/ou 'artist'."})

    by_genre = songs["genre_key"] == genre_key
    by_artist = songs["artist_key"] == artist_key
    warnings = []
    if genre_key is not None and not by_genre.any():
        warnings.append({"message": f"Gênero '{request.genre}' não encontrado.",
                         "suggestions": suggest(request.genre, df["genre"])})
    if artist_key is not None and not by_artist.any():
        warnings.append({"message": f"Artista '{request.artist}' não encontrado.",
                         "suggestions": suggest(request.artist, df["artist"])})

    matches = songs[by_genre | by_artist].copy()
    if matches.empty:
        raise HTTPException(status_code=404, detail={
            "message": "Nenhuma música encontrada para o gênero/artista informado.",
            "details": warnings,
        })
    matches["match"] = np.select(
        [by_genre[matches.index] & by_artist[matches.index], by_genre[matches.index]],
        ["genre+artist", "genre"],
        default="artist",
    )
    top = matches.sort_values("Popularity", ascending=False, kind="mergesort").head(request.limit)

    response = {
        "genre": request.genre,
        "artist": request.artist,
        "total_matches": len(matches),
        "total": len(top),
        "recommendations": [{**song_info(row), "match": row["match"]} for _, row in top.iterrows()],
    }
    if warnings:
        response["warnings"] = warnings
    return response


@app.get("/recommendations/collaborative/{user_id}", tags=["Recomendações"],
         summary="3. Filtro colaborativo")
async def collaborative_recommendations(
    user_id: str,
    limit: int = Query(5, ge=1, le=100, description="Número de recomendações a retornar"),
):
    """
    Recomendações baseadas no comportamento de **usuários similares**, com dados fictícios
    de "quem gostou disso também gostou daquilo" (usuários `user1` a `user15`, veja `GET /users`).

    Lógica de **contagem de co-ocorrências**: para cada par de músicas contamos quantos
    usuários gostaram das duas. O `score` de uma música que o usuário ainda não curtiu é a
    soma das co-ocorrências dela com as músicas que ele curtiu. Empates são decididos pela
    popularidade. `because_you_liked` mostra quais músicas curtidas geraram a recomendação.

    **Exemplo de requisição**

        GET /recommendations/collaborative/user1?limit=3

    **Exemplo de resposta** (`liked_songs` resumido)

        {
          "user_id": "user1",
          "liked_songs": [{"title": "Roar", "artist": "Katy Perry"}, "..."],
          "similar_users": [{"user_id": "user2", "songs_in_common": 3},
                            {"user_id": "user3", "songs_in_common": 2}],
          "total": 3,
          "recommendations": [
            {"title": "Die Young", "artist": "Kesha", "genre": "dance pop", "year": 2013,
             "popularity": 75, "score": 5, "because_you_liked": ["Roar", "Dark Horse", "Bad Romance", "TiK ToK", "Telephone"]},
            {"title": "Part Of Me", "artist": "Katy Perry", "genre": "dance pop", "year": 2012,
             "popularity": 71, "score": 3, "because_you_liked": ["Roar", "Bad Romance", "Telephone"]},
            {"title": "Diamonds", "artist": "Rihanna", "genre": "barbadian pop", "year": 2012,
             "popularity": 61, "score": 3, "because_you_liked": ["Roar", "Bad Romance", "Telephone"]}
          ]
        }
    """
    user = resolve_user(user_id)
    liked = set(user_song_keys[user])
    scores, reasons = collaborative_scores(user)
    ranked = sorted(scores, key=lambda key: (-scores[key], -int(df.at[canonical_index[key], "Popularity"])))

    similar_users = []
    for other, keys in user_song_keys.items():
        common = len(liked & set(keys))
        if other != user and common > 0:
            similar_users.append({"user_id": other, "songs_in_common": common})
    similar_users.sort(key=lambda item: -item["songs_in_common"])

    recommendations = []
    for key in ranked[:limit]:
        row = df.loc[canonical_index[key]]
        recommendations.append({
            **song_info(row),
            "score": int(scores[key]),
            "because_you_liked": [title_of(liked_key) for liked_key in reasons[key]],
        })

    return {
        "user_id": user,
        "liked_songs": [short_info(key) for key in user_song_keys[user]],
        "similar_users": similar_users,
        "total": len(recommendations),
        "recommendations": recommendations,
    }


@app.post("/recommendations/hybrid", tags=["Recomendações"], summary="4. Recomendação híbrida")
async def hybrid_recommendations(request: HybridRequest):
    """
    Combina a recomendação **baseada em conteúdo** (similaridade com `song_title`) com o
    **filtro colaborativo** (co-ocorrências do `user_id`):

        hybrid_score = content_weight * content_score + collab_weight * collab_score

    - Os dois scores são normalizados entre 0 e 1 antes da combinação.
    - Os pesos são normalizados para somar 1 (ex.: 2 e 1 viram 0.667 e 0.333).
    - Não recomenda a própria música nem músicas que o usuário já curtiu.

    **Exemplo de requisição**

        POST /recommendations/hybrid
        {"song_title": "Shape of You", "user_id": "user12", "content_weight": 0.7,
         "collab_weight": 0.3, "limit": 5}

    **Exemplo de resposta**

        {
          "song": {"title": "Shape of You", "artist": "Ed Sheeran", "genre": "pop", "year": 2017,
                   "popularity": 87},
          "user_id": "user12",
          "weights": {"content_weight": 0.7, "collab_weight": 0.3},
          "total": 5,
          "recommendations": [
            {"title": "I'm Not The Only One", "artist": "Sam Smith", "genre": "pop", "year": 2015,
             "popularity": 84, "content_score": 0.9704, "collab_score": 1.0, "hybrid_score": 0.9793},
            {"title": "Let Her Go", "artist": "Passenger", "genre": "folk-pop", "year": 2014,
             "popularity": 77, "content_score": 0.8204, "collab_score": 1.0, "hybrid_score": 0.8742},
            {"title": "All I Ask", "artist": "Adele", "genre": "british soul", "year": 2016,
             "popularity": 71, "content_score": 0.7159, "collab_score": 1.0, "hybrid_score": 0.8012},
            {"title": "Closer", "artist": "The Chainsmokers", "genre": "electropop", "year": 2017,
             "popularity": 86, "content_score": 1.0, "collab_score": 0.0, "hybrid_score": 0.7},
            {"title": "Rockabye (feat. Sean Paul & Anne-Marie)", "artist": "Clean Bandit",
             "genre": "dance pop", "year": 2017, "popularity": 78, "content_score": 0.9943,
             "collab_score": 0.0, "hybrid_score": 0.696}
          ]
        }
    """
    index = find_song_index(request.song_title)
    user = resolve_user(request.user_id)
    total_weight = request.content_weight + request.collab_weight
    if total_weight <= 0:
        raise HTTPException(status_code=400, detail={
            "message": "content_weight + collab_weight deve ser maior que zero.",
        })
    content_weight = request.content_weight / total_weight
    collab_weight = request.collab_weight / total_weight

    seed = df.loc[index]
    excluded = set(user_song_keys[user]) | {seed["song_key"]}
    candidates = songs[~songs["song_key"].isin(excluded)].copy()

    content = similarity_matrix[index][candidates.index]
    content_range = content.max() - content.min()
    content_score = (content - content.min()) / content_range if content_range > 0 else np.zeros(len(content))

    scores, _ = collaborative_scores(user)
    collab = np.array([scores.get(key, 0) for key in candidates["song_key"]], dtype=float)
    collab_score = collab / collab.max() if collab.max() > 0 else np.zeros(len(collab))

    candidates["content_score"] = content_score
    candidates["collab_score"] = collab_score
    candidates["hybrid_score"] = content_weight * content_score + collab_weight * collab_score
    top = candidates.sort_values(["hybrid_score", "Popularity"], ascending=False, kind="mergesort").head(request.limit)

    return {
        "song": song_info(seed),
        "user_id": user,
        "weights": {"content_weight": round(content_weight, 4), "collab_weight": round(collab_weight, 4)},
        "total": len(top),
        "recommendations": [
            {
                **song_info(row),
                "content_score": round(float(row["content_score"]), 4),
                "collab_score": round(float(row["collab_score"]), 4),
                "hybrid_score": round(float(row["hybrid_score"]), 4),
            }
            for _, row in top.iterrows()
        ],
    }


@app.get("/recommendations/popular", tags=["Recomendações"], summary="5. Recomendação por popularidade/ano")
async def popular_recommendations(
    year: Optional[int] = Query(None, description="Filtrar por ano (2010 a 2019)"),
    genre: Optional[str] = Query(None, description="Filtrar por gênero (ex.: 'dance pop')"),
    limit: int = Query(5, ge=1, le=100, description="Número de resultados"),
):
    """
    Retorna as músicas mais populares, opcionalmente filtradas por `year` e/ou `genre`,
    ordenadas por popularidade (maior primeiro). Sem filtros, considera o dataset inteiro.

    **Exemplo de requisição**

        GET /recommendations/popular?year=2015&genre=dance%20pop&limit=3

    **Exemplo de resposta**

        {
          "filters": {"year": 2015, "genre": "dance pop"},
          "total": 3,
          "recommendations": [
            {"title": "Uptown Funk", "artist": "Mark Ronson", "genre": "dance pop", "year": 2015,
             "popularity": 82},
            {"title": "Love Me Like You Do - From \\"Fifty Shades Of Grey\\"", "artist": "Ellie Goulding",
             "genre": "dance pop", "year": 2015, "popularity": 79},
            {"title": "Want to Want Me", "artist": "Jason Derulo", "genre": "dance pop", "year": 2015,
             "popularity": 77}
          ]
        }
    """
    data = df
    if year is not None:
        data = data[data["year"] == year]
        if data.empty:
            raise HTTPException(status_code=404, detail={
                "message": f"Não há músicas do ano {year} no dataset.",
                "available_years": sorted(int(y) for y in df["year"].unique()),
            })
    if genre and genre.strip():
        genre_key = normalize(genre)
        if not (df["genre_key"] == genre_key).any():
            raise HTTPException(status_code=404, detail={
                "message": f"Gênero '{genre}' não encontrado.",
                "suggestions": suggest(genre, df["genre"]),
            })
        data = data[data["genre_key"] == genre_key]
        if data.empty:
            raise HTTPException(status_code=404, detail={
                "message": f"Nenhuma música do gênero '{genre}' no ano {year}.",
            })

    top = (data.sort_values("Popularity", ascending=False, kind="mergesort")
               .drop_duplicates("song_key")
               .head(limit))
    return {
        "filters": {"year": year, "genre": genre},
        "total": len(top),
        "recommendations": [song_info(row) for _, row in top.iterrows()],
    }
