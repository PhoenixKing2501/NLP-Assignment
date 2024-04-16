import re
from collections import Counter
from typing import List, Union

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# import nltk
# nltk.download('punkt')
# nltk.download('stopwords')


def preprocess_text(text: str,
                    remove_stopwords: bool = True) -> List[str]:
    # Lowercasing
    text = text.lower()

    # Replace '\\' with space
    text = text.replace('\\', ' ')

    # Remove unnecessary symbols and non-alphanumeric characters
    # if a number like 123,456,789 is present, change it to 123456789
    text = re.sub(r'(\d),(\d)', r'\1\2', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

    # Tokenization
    tokens = word_tokenize(text)

    if not remove_stopwords:
        return tokens

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token
              for token in tokens
              if token not in stop_words]

    return tokens
# END preprocess_text


def build_vocabulary(
    corpus: List[List[str]],
    min_df: Union[int, float] = 1,
    max_df: Union[int, float] = 1.0,
):
    if isinstance(min_df, float):
        min_df = int(min_df * len(corpus))

    if isinstance(max_df, float):
        max_df = int(max_df * len(corpus))

    # Count word occurrences in documents
    word_counts: Counter[str] = Counter()
    for document in corpus:
        word_counts.update(set(document))

    # Filter based on document frequency
    filtered_words = [word
                      for word, count in word_counts.items()
                      if min_df <= count <= max_df]

    # Create vocabulary
    vocabulary = {word: index
                  for index, word in enumerate(filtered_words)}
    return vocabulary
# END build_vocabulary


def main() -> None:
    # Example corpus of texts
    corpus = [
        "This is a test sentence.",
        "Another test sentence here.",
        "Yet another sentence for testing purposes.",
        "This sentence contains some unnecessary symbols and stop words.",
    ]

    # Preprocess the corpus
    preprocessed_corpus = [preprocess_text(sentence)
                           for sentence in corpus]

    # Build vocabulary
    vocabulary = build_vocabulary(preprocessed_corpus,
                                  min_df=1, max_df=0.75)

    print("Vocabulary:")
    print(vocabulary)
# END main


if __name__ == "__main__":
    main()
