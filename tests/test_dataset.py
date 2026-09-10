from app.dataset import _match_query


def test_known_query_returns_expected_doc(dataset_search):
    hits = dataset_search.search("Batman fights the Joker in Gotham", k=3)
    ids = {d.doc_id for d in hits}
    assert "doc_001" in ids  # The Dark Knight
    assert hits[0].score <= hits[-1].score  # bm25: lower is better, sorted asc


def test_search_ranks_relevant_doc_first(dataset_search):
    hits = dataset_search.search("dream heist planting an idea in the subconscious", k=3)
    assert hits[0].doc_id == "doc_017"  # Inception


def test_no_match_returns_empty(dataset_search):
    assert dataset_search.search("quantum chromodynamics lattice gauge theory") == []


def test_query_of_only_stopwords_returns_empty(dataset_search):
    assert dataset_search.search("what is the a of and or") == []


def test_match_query_drops_stopwords_and_quotes_terms():
    q = _match_query("Who is the villain in Black Panther?")
    assert q == '"villain" OR "black" OR "panther"'
