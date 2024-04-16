from typing import List
import pandas
from spacy.tokens import Token, Doc
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


def preprocess(
    text: str,
) -> str:
    return ''.join(c for c in text if c.isalnum() or c.isspace())
# END def preprocess


def correct_spellings(
    doc: Doc,
    debug: bool
) -> str:
    """
    Correct the spellings of the text using the spaCy language model.

    Args:
        doc (Doc): The spaCy document to correct the spellings of.
        debug (bool): Whether to print the original and corrected text.

    Returns:
        str: The corrected text.
    """

    text = doc.text
    if doc._.performed_spellCheck:
        out = doc._.outcome_spellCheck
        if debug:
            print(f"Original : {text}")
            print(f"Corrected: {out}")
            print()
        # END if debug

        return out
    # END if performed_spellCheck

    return text
# END def correctSpellings


def perform_ner_pos(
    doc: Doc,
):
    """
    Perform Named Entity Recognition (NER) and Part-of-Speech (POS) tagging on the text.

    Args:
        doc (Doc): The spaCy document to perform NER and POS tagging on.

    Returns:
        The named entities and POS tags.
    """

    # Extract named entities
    named_entities = [f'{token.lemma_}_NAMED_ENTITY'
                      for ent in doc.ents
                      for token in ent]

    def check(token: Token) -> bool:
        return token.ent_type == 0 \
            and token.ent_iob in [0, 2] \
            and token.text.isalnum()

    # Extract POS tags
    pos_tags = [f'{token.lemma_}_{token.pos_}'
                for token in doc if check(token)]

    return named_entities, pos_tags
# END def perform_ner_pos


def get_ner_pos(
    doc: Doc,
):
    """
    Get the named entities and POS tags from the spaCy document.

    Args:
        doc (Doc): The spaCy document to get the named entities and POS tags from.

    Returns:
        The named entities and POS tags.
    """

    named_entities, pos_tags = perform_ner_pos(doc)
    return ' '.join(named_entities + pos_tags)
# END def get_ner_pos


def get_vectors(
    docs: List[str],
    queries: List[str],
    add_noise: bool = False
):
    """
    Get the vectors for the documents and queries.

    Args:
        docs (List[str]): The documents to get the vectors for.
        queries (List[str]): The queries to get the vectors for.

    Returns:
        np.ndarray: The vectors for the documents.
        np.ndarray: The vectors for the queries.
        dict: The vocabulary of the vectorizer.
    """

    # Get the vectors for the documents and queries

    vectorizer = TfidfVectorizer(min_df=5, max_df=0.85,
                                 max_features=10000)

    docVectors = vectorizer.fit_transform(docs)
    queryVectors = vectorizer.transform(queries)

    docVectors = np.array(docVectors.todense())  # type: ignore
    queryVectors = np.array(queryVectors.todense())  # type: ignore

    # Add Gaussian Noise to the vectors (mu = 0, sigma = 0.01)
    if add_noise:
        gaussian_noise = np.random.normal(0, 0.01, docVectors.shape)
        docVectors += gaussian_noise

        gaussian_noise = np.random.normal(0, 0.01, queryVectors.shape)
        queryVectors += gaussian_noise
    # END if add_noise

    return docVectors, queryVectors, vectorizer.vocabulary_
# END def get_vectors


def calculate_precision_at_k(
    docs: pandas.DataFrame,
    queries: pandas.DataFrame,
    cosine_similarities: np.ndarray,
    k: int
):
    """
    Calculate the precision at k for the given documents and queries.

    Args:
        docs (pandas.DataFrame): The documents to calculate the precision at k for.
        queries (pandas.DataFrame): The queries to calculate the precision at k for.
        cosine_similarities (np.ndarray): The cosine similarities between the queries and documents.
        k (int): The value of k to calculate the precision at.

    Returns:
        float: The precision at k.
    """
    precision_scores = []
    for i in range(len(queries)):
        query_similarities = cosine_similarities[i]
        sorted_indices = np.argsort(query_similarities)[::-1]
        top_k_indices = sorted_indices[:k]
        relevant_docs = [doc for j, doc in enumerate(docs['doc_id'])
                         if j in top_k_indices]
        q_relevant_docs = queries['relevant_docs'][i]
        n = min(k, len(q_relevant_docs))
        precision = len(set(relevant_docs) & set(q_relevant_docs)) / n
        precision_scores.append(precision)
    # END for i in range(len(queries))

    return np.mean(precision_scores)
# END def calculate_precision_at_k


def print_scores(
    docs: pandas.DataFrame,
    queries: pandas.DataFrame,
    docVectors: np.ndarray,
    queryVectors: np.ndarray,
):
    cosine_similarities = cosine_similarity(queryVectors, docVectors)

    for k in [1, 5, 10]:
        precision = calculate_precision_at_k(docs, queries,
                                             cosine_similarities, k)
        print(f'Precision@{k: <2}: {precision:.4f}')
    # END for k in [1, 5, 10]

# END def print_scores
