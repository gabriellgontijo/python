"""
Testes das 5 APIs de recomendação.

Executar:  python -m pytest -v test_main.py
"""

from urllib.parse import quote

from fastapi.testclient import TestClient

from main import app, df, features, songs, user_likes

client = TestClient(app)


def content_url(title: str) -> str:
    return "/recommendations/content-based/" + quote(title, safe="")


def titles(response) -> list:
    return [item["title"] for item in response.json()["recommendations"]]


# ---------------------------------------------------------------------------
# Dados e pré-processamento
# ---------------------------------------------------------------------------

def test_dataset_preprocessado():
    assert set(features + ["title", "artist", "genre", "year"]).issubset(df.columns)
    assert len(df) == 602                      # 603 linhas - 1 registro inválido (BPM 0)
    assert (df["Beats.Per.Minute"] > 0).all()
    assert "Señorita" in set(df["title"])      # caracteres corrigidos ("Se駉rita")
    assert songs["song_key"].is_unique


def test_raiz_e_usuarios():
    response = client.get("/")
    assert response.status_code == 200
    assert len(response.json()["endpoints"]) == 5

    response = client.get("/users")
    assert response.status_code == 200
    assert set(response.json()["users"]) == set(user_likes)


def test_rotas_do_enunciado():
    paths = client.get("/openapi.json").json()["paths"]
    assert "get" in paths["/recommendations/content-based/{song_title}"]
    assert "post" in paths["/recommendations/genre-artist"]
    assert "get" in paths["/recommendations/collaborative/{user_id}"]
    assert "post" in paths["/recommendations/hybrid"]
    assert "get" in paths["/recommendations/popular"]


# ---------------------------------------------------------------------------
# 1. Baseada em conteúdo
# ---------------------------------------------------------------------------

def test_conteudo_padrao():
    response = client.get(content_url("Hey, Soul Sister"))
    assert response.status_code == 200
    data = response.json()
    assert data["song"]["title"] == "Hey, Soul Sister"
    assert data["total"] == 5                                     # limit padrão
    assert "Hey, Soul Sister" not in titles(response)             # não recomenda a própria música
    similarities = [item["similarity"] for item in data["recommendations"]]
    assert similarities == sorted(similarities, reverse=True)
    assert all(0 <= value <= 1 for value in similarities)


def test_conteudo_limit_e_sem_repeticao():
    response = client.get(content_url("Love Yourself"), params={"limit": 50})
    assert response.status_code == 200
    items = response.json()["recommendations"]
    assert len(items) == 50
    assert len({(item["title"], item["artist"]) for item in items}) == 50


def test_conteudo_weights_json_e_formato_curto():
    sem_pesos = client.get(content_url("Shape of You"), params={"limit": 10})
    com_pesos = client.get(content_url("Shape of You"),
                           params={"limit": 10, "weights": '{"Beats.Per.Minute": 10, "Energy": 0}'})
    assert com_pesos.status_code == 200
    assert com_pesos.json()["weights"]["Beats.Per.Minute"] == 10
    assert com_pesos.json()["weights"]["Energy"] == 0
    assert com_pesos.json()["weights"]["Danceability"] == 1       # não informado -> peso 1
    assert titles(com_pesos) != titles(sem_pesos)                  # os pesos mudam o resultado

    curto = client.get(content_url("Shape of You"), params={"limit": 10, "weights": "bpm:10,energy:0"})
    assert curto.status_code == 200
    assert titles(curto) == titles(com_pesos)


def test_conteudo_weights_invalidos():
    for weights in ['{"Energia": 2}', '{"Energy": -1}', "abc", '{"Energy": "x"}', "[1, 2]",
                    '{"' + '": 0, "'.join(features) + '": 0}']:
        response = client.get(content_url("Shape of You"), params={"weights": weights})
        assert response.status_code == 400, weights


def test_conteudo_busca_flexivel():
    assert client.get(content_url("hey, soul sister")).json()["song"]["title"] == "Hey, Soul Sister"
    assert client.get(content_url("senorita")).json()["song"]["title"] == "Señorita"
    assert client.get(content_url("Titanium")).json()["song"]["title"] == "Titanium (feat. Sia)"
    # título com "/" funciona na URL
    response = client.get(content_url('Let It Go - From "Frozen / Single Version'))
    assert response.status_code == 200


def test_conteudo_titulo_repetido_com_artista():
    assert client.get(content_url("Hello")).json()["song"]["artist"] == "Adele"   # mais popular
    response = client.get(content_url("Hello"), params={"artist": "Martin Solveig"})
    assert response.json()["song"]["artist"] == "Martin Solveig"


def test_conteudo_erros():
    response = client.get(content_url("Musica Que Nao Existe"))
    assert response.status_code == 404
    assert "suggestions" in response.json()["detail"]
    assert client.get(content_url("Shape of You"), params={"limit": 0}).status_code == 422
    assert client.get(content_url("Shape of You"), params={"limit": 101}).status_code == 422


# ---------------------------------------------------------------------------
# 2. Gênero/Artista
# ---------------------------------------------------------------------------

def test_genero():
    response = client.post("/recommendations/genre-artist", json={"genre": "Dance Pop", "limit": 10})
    assert response.status_code == 200
    items = response.json()["recommendations"]
    assert len(items) == 10
    assert all(item["genre"] == "dance pop" and item["match"] == "genre" for item in items)
    popularity = [item["popularity"] for item in items]
    assert popularity == sorted(popularity, reverse=True)


def test_artista():
    response = client.post("/recommendations/genre-artist", json={"artist": "adele", "limit": 50})
    assert response.status_code == 200
    items = response.json()["recommendations"]
    assert items and all(item["artist"] == "Adele" for item in items)
    assert len({item["title"] for item in items}) == len(items)   # sem repetição de anos


def test_genero_ou_artista():
    response = client.post("/recommendations/genre-artist",
                           json={"genre": "canadian pop", "artist": "Ed Sheeran", "limit": 100})
    data = response.json()
    assert response.status_code == 200
    assert {item["match"] for item in data["recommendations"]} == {"genre", "artist"}
    assert all(item["genre"] == "canadian pop" or item["artist"] == "Ed Sheeran"
               for item in data["recommendations"])
    popularity = [item["popularity"] for item in data["recommendations"]]
    assert popularity == sorted(popularity, reverse=True)


def test_genero_artista_erros():
    assert client.post("/recommendations/genre-artist", json={}).status_code == 400
    assert client.post("/recommendations/genre-artist", json={"genre": "xyz", "artist": "xyz"}).status_code == 404
    assert client.post("/recommendations/genre-artist", json={"genre": "pop", "limit": 0}).status_code == 422
    # um filtro válido e outro inexistente: devolve o válido e avisa
    response = client.post("/recommendations/genre-artist", json={"genre": "xyz", "artist": "Adele"})
    assert response.status_code == 200
    assert "warnings" in response.json()


# ---------------------------------------------------------------------------
# 3. Filtro colaborativo
# ---------------------------------------------------------------------------

def test_colaborativo():
    response = client.get("/recommendations/collaborative/user1")
    assert response.status_code == 200
    data = response.json()
    liked = {item["title"] for item in data["liked_songs"]}
    assert data["total"] == 5
    assert not liked & set(titles(response))                     # não recomenda o que já curtiu
    scores = [item["score"] for item in data["recommendations"]]
    assert scores == sorted(scores, reverse=True)
    assert data["similar_users"][0] == {"user_id": "user2", "songs_in_common": 3}


def test_colaborativo_contagem_de_coocorrencias():
    # Conferência independente: score(m) = soma, para cada outro usuário que curtiu m,
    # do número de músicas em comum com o usuário.
    liked = set(user_likes["user1"])
    expected = {}
    for other, other_likes in user_likes.items():
        if other == "user1":
            continue
        common = len(liked & set(other_likes))
        for title in other_likes:
            if title not in liked and common:
                expected[title] = expected.get(title, 0) + common
    response = client.get("/recommendations/collaborative/user1", params={"limit": 100})
    got = {item["title"]: item["score"] for item in response.json()["recommendations"]}
    assert got == expected


def test_colaborativo_ids_e_erros():
    assert client.get("/recommendations/collaborative/1").json()["user_id"] == "user1"
    assert client.get("/recommendations/collaborative/USER2").json()["user_id"] == "user2"
    response = client.get("/recommendations/collaborative/user999")
    assert response.status_code == 404
    assert "available_users" in response.json()["detail"]


# ---------------------------------------------------------------------------
# 4. Híbrida
# ---------------------------------------------------------------------------

def hybrid(**body):
    payload = {"song_title": "Shape of You", "user_id": "user12"}
    payload.update(body)
    return client.post("/recommendations/hybrid", json=payload)


def test_hibrida_padrao():
    response = hybrid()
    assert response.status_code == 200
    data = response.json()
    assert data["weights"] == {"content_weight": 0.7, "collab_weight": 0.3}
    assert data["total"] == 5
    liked = set(user_likes["user12"])
    for item in data["recommendations"]:
        assert item["title"] != "Shape of You" and item["title"] not in liked
        expected = 0.7 * item["content_score"] + 0.3 * item["collab_score"]
        assert abs(item["hybrid_score"] - expected) < 1e-3
    scores = [item["hybrid_score"] for item in data["recommendations"]]
    assert scores == sorted(scores, reverse=True)
    # mistura as duas abordagens: há músicas vindas do colaborativo e só do conteúdo
    assert any(item["collab_score"] > 0 for item in data["recommendations"])
    assert any(item["collab_score"] == 0 for item in data["recommendations"])


def test_hibrida_pesos_extremos():
    so_conteudo = hybrid(content_weight=1, collab_weight=0, limit=10)
    conteudo = client.get(content_url("Shape of You"), params={"limit": 20})
    liked = set(user_likes["user12"])
    esperado = [title for title in titles(conteudo) if title not in liked][:10]
    assert titles(so_conteudo) == esperado

    so_colab = hybrid(content_weight=0, collab_weight=1, limit=3)
    assert all(item["collab_score"] == 1.0 for item in so_colab.json()["recommendations"])

    # pesos são normalizados para somar 1
    assert hybrid(content_weight=2, collab_weight=1).json()["weights"] == {
        "content_weight": 0.6667, "collab_weight": 0.3333}


def test_hibrida_erros():
    assert hybrid(content_weight=0, collab_weight=0).status_code == 400
    assert hybrid(content_weight=-1).status_code == 422
    assert hybrid(song_title="Musica Que Nao Existe").status_code == 404
    assert hybrid(user_id="user999").status_code == 404
    assert client.post("/recommendations/hybrid", json={"song_title": "Roar"}).status_code == 422


# ---------------------------------------------------------------------------
# 5. Popularidade/Ano
# ---------------------------------------------------------------------------

def test_popular_sem_filtros():
    response = client.get("/recommendations/popular")
    assert response.status_code == 200
    items = response.json()["recommendations"]
    assert len(items) == 5
    assert items[0]["popularity"] == int(df["Popularity"].max())
    popularity = [item["popularity"] for item in items]
    assert popularity == sorted(popularity, reverse=True)


def test_popular_filtros():
    response = client.get("/recommendations/popular", params={"year": 2015, "genre": "dance pop", "limit": 10})
    assert response.status_code == 200
    items = response.json()["recommendations"]
    assert len(items) == 10
    assert all(item["year"] == 2015 and item["genre"] == "dance pop" for item in items)

    response = client.get("/recommendations/popular", params={"year": 2016, "limit": 100})
    items = response.json()["recommendations"]
    assert all(item["year"] == 2016 for item in items)
    assert len({(item["title"], item["artist"]) for item in items}) == len(items)


def test_popular_erros():
    assert client.get("/recommendations/popular", params={"year": 2005}).status_code == 404
    assert client.get("/recommendations/popular", params={"genre": "xyz"}).status_code == 404
    assert client.get("/recommendations/popular", params={"year": 2010, "genre": "latin"}).status_code == 404
    assert client.get("/recommendations/popular", params={"limit": 0}).status_code == 422
