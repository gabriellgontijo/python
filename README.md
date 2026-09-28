# Sistema de Recomendação Musical com FastAPI

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gabriellgontijo/python/blob/main/Sistema_Recomendacao_Musical.ipynb)

Atividade Prática: sistema de recomendação musical com **5 APIs** em FastAPI, usando
abordagens baseadas em conteúdo, gênero/artista, filtro colaborativo, híbrida e popularidade.
Os dados vêm do arquivo `top50MusicFrom2010-2019.csv` (Top 50 do Spotify de 2010 a 2019),
disponibilizado pelo professor.

| # | Método | Endpoint | Abordagem |
|---|--------|----------|-----------|
| 1 | GET  | `/recommendations/content-based/{song_title}` | Similaridade do cosseno entre as características musicais |
| 2 | POST | `/recommendations/genre-artist` | Mesmo gênero **ou** artista, ordenado por popularidade |
| 3 | GET  | `/recommendations/collaborative/{user_id}` | Filtro colaborativo com contagem de co-ocorrências |
| 4 | POST | `/recommendations/hybrid` | Combinação ponderada de conteúdo + colaborativo |
| 5 | GET  | `/recommendations/popular` | Mais populares, com filtro por ano e/ou gênero |

## Arquivos

| Arquivo | Conteúdo |
|---------|----------|
| `Sistema_Recomendacao_Musical.ipynb` | Notebook para o Google Colab: instala, carrega os dados, cria a API, roda os testes e chama cada endpoint |
| `main.py` | Código da API (FastAPI) |
| `test_main.py` | 23 testes automatizados (pytest) cobrindo as 5 APIs e os casos de erro |
| `top50MusicFrom2010-2019.csv` | Dataset disponibilizado pelo professor |
| `requirements.txt` | Dependências |
| `Atividade Prática.pdf` | Enunciado |

## Como executar

### No Google Colab (recomendado)

1. Abra o notebook pelo botão **Open in Colab** acima (ou em *Arquivo → Fazer upload de notebook*
   e selecione `Sistema_Recomendacao_Musical.ipynb`).
2. Menu **Ambiente de execução → Executar tudo** (`Ctrl+F9`).

O notebook baixa o CSV sozinho. Se não houver internet, ele pede o upload do arquivo. No fim,
mostra um link para a documentação interativa (Swagger), para testar as APIs pelo navegador.

### Localmente

```bash
pip install -r requirements.txt
uvicorn main:app --reload          # API em http://127.0.0.1:8000
python -m pytest -v test_main.py   # testes
```

Documentação interativa: <http://127.0.0.1:8000/docs>. Cada endpoint traz a descrição e exemplos
de requisição e resposta.

## Pré-processamento dos dados

- **Nomes de coluna**: o CSV usa nomes longos (ex.: `Energy- The energy of a song - ...`). Eles
  são renomeados para os nomes do código base (`Beats.Per.Minute`, `Energy`, `Danceability`,
  `Loudness/dB`, `Liveness`, `Valence`, `Length`, `Acousticness`, `Speechiness`, `Popularity`),
  mais `title`, `artist`, `genre` e `year`.
- **Caracteres corrompidos**: 10 títulos vieram com acentos quebrados (ex.: `Se駉rita`,
  `Bon app閠it`) e são corrigidos para `Señorita` e `Bon appétit`.
- **Registro inválido**: `Million Years Ago` (Adele) tem BPM 0 e todas as características
  zeradas. Esse registro é removido, porque também distorceria a normalização. Restam 602 linhas.
- **Normalização**: `MinMaxScaler` nas 10 características, seguido da matriz de
  `cosine_similarity`, como no código base. A matriz normalizada fica separada, para que as
  respostas mostrem os valores originais.
- **Músicas repetidas**: 16 músicas aparecem duas vezes no CSV (ex.: `Love Yourself` em 2015 e
  2016). Nas recomendações cada música aparece uma única vez, na versão mais popular.
- **Busca de títulos**: não diferencia maiúsculas/minúsculas nem acentos, e aceita o título sem
  complemento (`Titanium` encontra `Titanium (feat. Sia)`). Se a música não existir, a resposta
  404 sugere títulos parecidos.

## Endpoints e exemplos

Todas as respostas abaixo foram geradas pela própria API.

### 1. Recomendação baseada em conteúdo

`GET /recommendations/content-based/{song_title}`

| Parâmetro | Tipo | Padrão | Descrição |
|-----------|------|--------|-----------|
| `limit` | int (1–100) | 5 | Número de recomendações |
| `weights` | dicionário (JSON) | todos 1 | Peso de cada característica, ex.: `{"Energy": 2, "Danceability": 1.5}` |
| `artist` | texto | — | Opcional, para títulos repetidos entre artistas (ex.: `Hello`) |

A similaridade é o cosseno entre as características normalizadas. Com `weights`, usa-se o
cosseno ponderado, `Σ w·x·y / (√Σ w·x² · √Σ w·y²)`. Características não informadas ficam com
peso 1, e peso 0 ignora a característica.

> **Por que `weights` é um JSON na URL?** Na assinatura do código base
> (`weights: Optional[Dict[str, float]]`), o FastAPI trata um `Dict` como **corpo** da
> requisição. Em um GET, o Swagger não envia corpo, e `?weights=...` na URL é ignorado (vira `None`).
> Por isso o parâmetro é recebido como texto e convertido para dicionário. Também aceita o formato
> curto `Energy:2,Danceability:1.5`.

```bash
curl -G "http://127.0.0.1:8000/recommendations/content-based/Shape%20of%20You" \
     --data-urlencode "limit=2" \
     --data-urlencode 'weights={"Energy": 2, "Danceability": 2}'
```

```json
{
  "song": {
    "title": "Shape of You", "artist": "Ed Sheeran", "genre": "pop", "year": 2017, "popularity": 87,
    "features": {"Beats.Per.Minute": 96, "Energy": 65, "Danceability": 83, "Loudness/dB": -3, "Liveness": 9,
                 "Valence": 93, "Length": 234, "Acousticness": 58, "Speechiness": 8, "Popularity": 87}
  },
  "weights": {"Beats.Per.Minute": 1.0, "Energy": 2.0, "Danceability": 2.0, "Loudness/dB": 1.0, "Liveness": 1.0,
              "Valence": 1.0, "Length": 1.0, "Acousticness": 1.0, "Speechiness": 1.0, "Popularity": 1.0},
  "total": 2,
  "recommendations": [
    {"title": "Closer", "artist": "The Chainsmokers", "genre": "electropop", "year": 2017, "popularity": 86,
     "similarity": 0.9901,
     "features": {"Beats.Per.Minute": 95, "Energy": 52, "Danceability": 75, "Loudness/dB": -6, "Liveness": 11,
                  "Valence": 66, "Length": 245, "Acousticness": 41, "Speechiness": 3, "Popularity": 86}},
    {"title": "Sing", "artist": "Ed Sheeran", "genre": "pop", "year": 2015, "popularity": 71,
     "similarity": 0.9872,
     "features": {"Beats.Per.Minute": 120, "Energy": 67, "Danceability": 82, "Loudness/dB": -4, "Liveness": 6,
                  "Valence": 94, "Length": 235, "Acousticness": 30, "Speechiness": 5, "Popularity": 71}}
  ]
}
```

Música inexistente (`GET /recommendations/content-based/Shap%20of%20Yuo`) → **404**:

```json
{"detail": {"message": "Música 'Shap of Yuo' não encontrada.",
            "suggestions": ["Shape of You", "Shake It Off", "Hall of Fame", "What Do You Mean?", "There for You"]}}
```

### 2. Recomendação por gênero/artista

`POST /recommendations/genre-artist`, com corpo `{"genre": ..., "artist": ..., "limit": 5}`.

Retorna as músicas do gênero **ou** do artista informado, da mais popular para a menos popular.
Pode-se informar só `genre`, só `artist` ou os dois. O campo `match` indica o motivo:
`genre`, `artist` ou `genre+artist`.

```bash
curl -X POST "http://127.0.0.1:8000/recommendations/genre-artist" \
     -H "Content-Type: application/json" \
     -d '{"genre": "canadian pop", "artist": "Ed Sheeran", "limit": 3}'
```

```json
{
  "genre": "canadian pop",
  "artist": "Ed Sheeran",
  "total_matches": 42,
  "total": 3,
  "recommendations": [
    {"title": "Señorita", "artist": "Shawn Mendes", "genre": "canadian pop", "year": 2019, "popularity": 95, "match": "genre"},
    {"title": "South of the Border (feat. Camila Cabello & Cardi B)", "artist": "Ed Sheeran", "genre": "pop", "year": 2019, "popularity": 92, "match": "artist"},
    {"title": "Shape of You", "artist": "Ed Sheeran", "genre": "pop", "year": 2017, "popularity": 87, "match": "artist"}
  ]
}
```

Sem `genre` e sem `artist` → **400** `{"detail": {"message": "Informe 'genre' e/ou 'artist'."}}`.

### 3. Filtro colaborativo

`GET /recommendations/collaborative/{user_id}?limit=5`

- **Dados fictícios**: dicionário `user_likes` com 15 usuários (`user1` a `user15`) e as músicas
  que cada um curtiu. Os gostos formam grupos: pop dançante, pop/funk, EDM, pop canadense,
  baladas e latino. `GET /users` lista todos.
- **Co-ocorrências**: para cada par de músicas, conta-se quantos usuários gostaram das duas
  ("quem gostou disso também gostou daquilo"). O `score` de cada música ainda não curtida é a
  soma das co-ocorrências dela com as músicas que o usuário curtiu. Empates são decididos pela
  popularidade.
- `user_id` aceita `user1`, `USER1` ou apenas `1`.

```bash
curl "http://127.0.0.1:8000/recommendations/collaborative/user1?limit=3"
```

```json
{
  "user_id": "user1",
  "liked_songs": [
    {"title": "Roar", "artist": "Katy Perry"}, {"title": "Dark Horse", "artist": "Katy Perry"},
    {"title": "Bad Romance", "artist": "Lady Gaga"}, {"title": "TiK ToK", "artist": "Kesha"},
    {"title": "Only Girl (In The World)", "artist": "Rihanna"}, {"title": "Telephone", "artist": "Lady Gaga"}
  ],
  "similar_users": [{"user_id": "user2", "songs_in_common": 3}, {"user_id": "user3", "songs_in_common": 2}],
  "total": 3,
  "recommendations": [
    {"title": "Die Young", "artist": "Kesha", "genre": "dance pop", "year": 2013, "popularity": 75, "score": 5,
     "because_you_liked": ["Roar", "Dark Horse", "Bad Romance", "TiK ToK", "Telephone"]},
    {"title": "Part Of Me", "artist": "Katy Perry", "genre": "dance pop", "year": 2012, "popularity": 71, "score": 3,
     "because_you_liked": ["Roar", "Bad Romance", "Telephone"]},
    {"title": "Diamonds", "artist": "Rihanna", "genre": "barbadian pop", "year": 2012, "popularity": 61, "score": 3,
     "because_you_liked": ["Roar", "Bad Romance", "Telephone"]}
  ]
}
```

Usuário inexistente → **404** com a lista `available_users`.

### 4. Recomendação híbrida

`POST /recommendations/hybrid`, com corpo
`{"song_title": ..., "user_id": ..., "content_weight": 0.7, "collab_weight": 0.3, "limit": 5}`.

```
hybrid_score = content_weight × content_score + collab_weight × collab_score
```

- `content_score` é a similaridade com `song_title`, normalizada entre 0 e 1.
- `collab_score` é o score colaborativo do `user_id`, dividido pelo maior score.
- Os pesos são normalizados para somar 1 (ex.: 2 e 1 viram 0.6667 e 0.3333).
- Não recomenda a própria música nem músicas que o usuário já curtiu.

```bash
curl -X POST "http://127.0.0.1:8000/recommendations/hybrid" \
     -H "Content-Type: application/json" \
     -d '{"song_title": "Shape of You", "user_id": "user12", "content_weight": 0.7, "collab_weight": 0.3, "limit": 5}'
```

```json
{
  "song": {"title": "Shape of You", "artist": "Ed Sheeran", "genre": "pop", "year": 2017, "popularity": 87},
  "user_id": "user12",
  "weights": {"content_weight": 0.7, "collab_weight": 0.3},
  "total": 5,
  "recommendations": [
    {"title": "I'm Not The Only One", "artist": "Sam Smith", "genre": "pop", "year": 2015, "popularity": 84,
     "content_score": 0.9704, "collab_score": 1.0, "hybrid_score": 0.9793},
    {"title": "Let Her Go", "artist": "Passenger", "genre": "folk-pop", "year": 2014, "popularity": 77,
     "content_score": 0.8204, "collab_score": 1.0, "hybrid_score": 0.8742},
    {"title": "All I Ask", "artist": "Adele", "genre": "british soul", "year": 2016, "popularity": 71,
     "content_score": 0.7159, "collab_score": 1.0, "hybrid_score": 0.8012},
    {"title": "Closer", "artist": "The Chainsmokers", "genre": "electropop", "year": 2017, "popularity": 86,
     "content_score": 1.0, "collab_score": 0.0, "hybrid_score": 0.7},
    {"title": "Rockabye (feat. Sean Paul & Anne-Marie)", "artist": "Clean Bandit", "genre": "dance pop", "year": 2017,
     "popularity": 78, "content_score": 0.9943, "collab_score": 0.0, "hybrid_score": 0.696}
  ]
}
```

Com `content_weight=1, collab_weight=0`, o resultado é igual ao da API 1. Com
`content_weight=0, collab_weight=1`, é igual ao da API 3. Os testes verificam os dois casos.

### 5. Recomendação por popularidade/ano

`GET /recommendations/popular?year=&genre=&limit=5`. Todos os parâmetros são opcionais.

```bash
curl "http://127.0.0.1:8000/recommendations/popular?year=2015&genre=dance%20pop&limit=3"
```

```json
{
  "filters": {"year": 2015, "genre": "dance pop"},
  "total": 3,
  "recommendations": [
    {"title": "Uptown Funk", "artist": "Mark Ronson", "genre": "dance pop", "year": 2015, "popularity": 82},
    {"title": "Love Me Like You Do - From \"Fifty Shades Of Grey\"", "artist": "Ellie Goulding", "genre": "dance pop", "year": 2015, "popularity": 79},
    {"title": "Want to Want Me", "artist": "Jason Derulo", "genre": "dance pop", "year": 2015, "popularity": 77}
  ]
}
```

Sem filtros (`GET /recommendations/popular?limit=3`):

```json
{
  "filters": {"year": null, "genre": null},
  "total": 3,
  "recommendations": [
    {"title": "Memories", "artist": "Maroon 5", "genre": "pop", "year": 2019, "popularity": 99},
    {"title": "Lose You To Love Me", "artist": "Selena Gomez", "genre": "dance pop", "year": 2019, "popularity": 97},
    {"title": "Someone You Loved", "artist": "Lewis Capaldi", "genre": "pop", "year": 2019, "popularity": 96}
  ]
}
```

Um ano fora de 2010–2019 ou um gênero inexistente → **404**, com os anos disponíveis ou
sugestões de gênero.

## Checklist do enunciado

| Requisito | Status | Onde |
|-----------|:------:|------|
| 5 endpoints com as rotas e métodos pedidos | ✅ | `main.py`; teste `test_rotas_do_enunciado` |
| 1. Conteúdo: similaridade por características, `limit` (padrão 5) e `weights` | ✅ | `content_based_recommendations` |
| 2. Gênero/Artista: POST com `genre`, `artist`, `limit`, ordenado por popularidade | ✅ | `genre_artist_recommendations` |
| 3. Colaborativo: dados fictícios + contagem de co-ocorrências | ✅ | `user_likes`, `co_occurrence` |
| 4. Híbrida: POST com `song_title`, `user_id`, `content_weight` (0.7), `collab_weight` (0.3), `limit` | ✅ | `hybrid_recommendations` |
| 5. Popularidade/Ano: GET com `year`, `genre`, `limit` opcionais | ✅ | `popular_recommendations` |
| Estrutura do código base (FastAPI, Pydantic, pandas, `MinMaxScaler`, `cosine_similarity`, `GenreArtistRequest`, `HybridRequest`) | ✅ | `main.py` |
| Simular dados de interação usuário-música | ✅ | `user_likes`, `GET /users` |
| Documentar cada endpoint com exemplos de requisição/resposta | ✅ | Este README, docstrings (Swagger `/docs`) e notebook |
| Testar todas as APIs localmente | ✅ | `test_main.py` (23 testes) e chamadas HTTP reais no notebook |
| Dicas: cosseno, dicionário de usuários, combinação de scores, ordenação por popularidade | ✅ | Todas aplicadas |
