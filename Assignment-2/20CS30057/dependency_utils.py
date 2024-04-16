from common import *

# set of all words in the corpus
all_words: List[str] = []


# set of all pos tags
all_pos_tags: List[str] = []


# set of all dependency labels
all_dependency_labels: List[str] = []


# most freq test pos pair dep relations ([pos, pos] -> dep
# when dep is most freq for that pos pair)
test_pos_pair_dep_relations: Dict[Tuple[str, str], str] = {}


def set_all_words(loader: Loader) -> None:
    """
    Calculate the counts of all words in the corpus.

    Only include the top 1000 most frequent normalized words in the corpus,
    which don't occur in more than 50% of the sentences.
    """

    _all_words: Dict[str, int] = {}

    for sentence in loader:
        words = {word.norm_word for word in sentence}

        for word in words:
            _all_words[word] = _all_words.get(word, 0) + 1
    # END for sentence in loader

    _all_words = {
        k: v
        for k, v in sorted(_all_words.items(),
                           key=lambda item: item[1],
                           reverse=True)
    }

    all_words.clear()

    for word, count in _all_words.items():
        if count > len(loader) / 2:
            continue

        all_words.append(word)
        if len(all_words) == 1000:
            break
    # END for word, count in _all_words.items()

# END set_all_words


def set_all_pos_tags(loader: Loader) -> None:
    """
    Calculate the counts of all pos tags in the corpus.
    """

    _all_pos_tags: Set[str] = set()

    for sentence in loader:
        for word in sentence:
            _all_pos_tags.add(word.pos_tag)
        # END for word in sentence
    # END for sentence in loader

    all_pos_tags.clear()
    all_pos_tags.extend(_all_pos_tags)
    all_pos_tags.sort()
# END set_all_pos_tags


def set_all_dependency_labels(loader: Loader) -> None:
    """
    Calculate the counts of all dependency labels in the corpus.
    """

    _all_dependency_labels: Set[str] = set()

    for sentence in loader:
        for word in sentence:
            _all_dependency_labels.add(word.dependency_label)
        # END for word in sentence
    # END for sentence in loader

    all_dependency_labels.clear()
    all_dependency_labels.extend(_all_dependency_labels)
    all_dependency_labels.sort()
# END set_all_dependency_labels


def set_test_pos_pair_dep_relations(loader: Loader) -> None:
    """
    Calculate the most frequent pos pair dep relations in the test set.
    """

    _test_pos_pair_dep_relations: Dict[Tuple[str, str], Dict[str, int]] = {}

    for sentence in loader:
        for word in sentence:
            # if the word is the root of the sentence, skip it
            if word.dependency_label == "root":
                continue

            key: Tuple[str, str] = (word.pos_tag,
                                    sentence[word.head_index].pos_tag)
            key_rev: Tuple[str, str] = (key[1], key[0])

            if key not in _test_pos_pair_dep_relations:
                _test_pos_pair_dep_relations[key] = {}

            _test_pos_pair_dep_relations[key][word.dependency_label] = \
                _test_pos_pair_dep_relations[key].get(
                    word.dependency_label, 0) + 1

            if key == key_rev:
                continue

            if key_rev not in _test_pos_pair_dep_relations:
                _test_pos_pair_dep_relations[key_rev] = {}

            _test_pos_pair_dep_relations[key_rev][word.dependency_label] = \
                _test_pos_pair_dep_relations[key_rev].get(
                    word.dependency_label, 0) + 1
        # END for word in sentence
    # END for sentence in loader

    test_pos_pair_dep_relations.clear()

    for key, value in _test_pos_pair_dep_relations.items():
        test_pos_pair_dep_relations[key] = \
            max(value, key=lambda k: value.get(k, -1))
# END set_test_pos_pair_dep_relations


def set_all(
    train_loader: Loader
) -> None:
    """
    Calculate the counts of all words, pos tags, and dependency labels in the corpus.
    """
    set_all_words(train_loader)
    set_all_pos_tags(train_loader)
    set_all_dependency_labels(train_loader)

    set_test_pos_pair_dep_relations(train_loader)
# END set_all


def main():
    train_loader = Loader(r"./data/train.txt")
    test_loader = Loader(r"./data/test.txt")

    set_all(train_loader)

    with open(fr"{SAVE_DIR}/dependency_all.txt",
              "w", encoding='utf-8') as file:
        print("all_words:\n", file=file)
        pp(all_words, stream=file)

        print("\n\nall_pos_tags:\n", file=file)
        pp(all_pos_tags, stream=file)

        print("\n\nall_dependency_labels:\n", file=file)
        pp(all_dependency_labels, stream=file)

        print("\n\ntest_pos_pair_dep_relations:\n", file=file)
        pp(test_pos_pair_dep_relations, stream=file)
# END main


if __name__ == "__main__":
    main()
