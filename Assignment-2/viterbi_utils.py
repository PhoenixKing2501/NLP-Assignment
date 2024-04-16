from common import *

# All states (POS tags)
all_states: List[str] = []

# All train wordset
all_train_words: List[str] = []

# All test wordset
all_test_words: List[str] = []

# Total sentences
total_sentences: int = 0

# Start counts of each state (POS tag)
# [state] -> counts
start_counts: Dict[str, int] = {}

# Total counts of each state (POS tag)
# [state] -> counts
state_counts: Dict[str, int] = {}

# Transition counts from one state to another
# [from_state, to_state] -> counts
transition_counts: Dict[Tuple[str, str], int] = {}

# Emission counts of a word given a state
# [word, state] -> counts
emission_counts: Dict[Tuple[str, str], int] = {}


def nlog_prob(prob: float) -> float:
    if prob == 0.0:
        return float("inf")

    return -1.0 * math.log(prob)
# END log_prob


def calculate_start_counts(loader: Loader):
    for sentence in loader:
        start_state = sentence[0].pos_tag

        start_counts[start_state] = start_counts.get(start_state, 0) + 1
    # END for sentence in loader
# END calculate_start_count


def calculate_state_counts(loader: Loader):
    for sentence in loader:
        for word in sentence:
            state = word.pos_tag

            state_counts[state] = state_counts.get(state, 0) + 1
        # END for word in sentence
    # END for sentence in loader

    global all_states
    all_states.extend(state_counts.keys())
    all_states.sort()
# END calculate_state_counts


def calculate_transition_counts(loader: Loader):
    for sentence in loader:
        for i in range(len(sentence) - 1):
            from_state = sentence[i].pos_tag
            to_state = sentence[i + 1].pos_tag

            transition_counts[from_state, to_state] = \
                transition_counts.get((from_state, to_state), 0) + 1
        # END for i in range
    # END for sentence in loader
# END calculate_transition_count


def calculate_emission_counts(loader: Loader):
    _all_words: Set[str] = set()

    for sentence in loader:
        for _word in sentence:
            word = _word.word
            # word = _word.norm_word
            state = _word.pos_tag

            _all_words.add(word)
            emission_counts[word, state] = \
                emission_counts.get((word, state), 0) + 1
        # END for _word in sentence
    # END for sentence in loader

    global all_train_words
    all_train_words.extend(_all_words)
    all_train_words.sort()
# END calculate_emission_count


def fill_test_words(loader: Loader):
    _all_words: Set[str] = set()

    for sentence in loader:
        for _word in sentence:
            word = _word.word
            # word = _word.norm_word

            _all_words.add(word)
        # END for _word in sentence
    # END for sentence in loader

    global all_test_words
    all_test_words.extend(_all_words)
    all_test_words.sort()
# END fill_test_words


def calculate_all_counts(
    train_loader: Loader,
    # test_loader: Loader
):
    global total_sentences
    if total_sentences == 0:
        total_sentences = len(train_loader)

        calculate_state_counts(train_loader)
        calculate_start_counts(train_loader)
        calculate_transition_counts(train_loader)
        calculate_emission_counts(train_loader)
        # fill_test_words(test_loader)
# END calculate_counts


def get_start_probability(
    state: str
) -> float:
    start_count = start_counts.get(state, 0)
    total_count = total_sentences

    if start_count == 0:
        prob = (start_count + 1) / (total_count + len(all_states))
    else:
        prob = start_count / total_count

    return nlog_prob(prob)
# END get_start_probability_smoothed


def get_transition_probability(
    from_state: str,
    to_state: str
) -> float:
    transition_count = transition_counts.get((from_state, to_state), 0)
    total_count = state_counts.get(from_state, 0)

    if transition_count == 0:
        prob = (transition_count + 1) / (total_count + len(all_states))
    else:
        prob = transition_count / total_count

    return nlog_prob(prob)
# END get_transition_probability_smoothed


def get_emission_probability(
    word: str,
    state: str
) -> float:
    emission_count = emission_counts.get((word, state), 0)
    total_count = state_counts.get(state, 0)

    if emission_count == 0:
        # prob = ((emission_count + 1) / (total_count +
        #                                 len(all_train_words) +
        #                                 len(all_test_words)))
        prob = ((emission_count + 1) / (total_count + len(all_train_words)))
    else:
        prob = emission_count / total_count

    return nlog_prob(prob)
# END get_emission_probability_smoothed


def save_counts():
    start_counts_file = f"{SAVE_DIR}/start_counts.tsv"
    state_counts_file = f"{SAVE_DIR}/state_counts.tsv"
    transition_counts_file = f"{SAVE_DIR}/transition_counts.tsv"
    emission_counts_file = f"{SAVE_DIR}/emission_counts.tsv"
    all_states_file = f"{SAVE_DIR}/all_states.txt"
    all_train_words_file = f"{SAVE_DIR}/all_train_words.txt"
    all_test_words_file = f"{SAVE_DIR}/all_test_words.txt"
    total_sentences_file = f"{SAVE_DIR}/total_sentences.txt"

    with open(start_counts_file, "w", encoding="utf-8") as f:
        for state, count in start_counts.items():
            f.write(f"{state}\t{count}\n")

    with open(state_counts_file, "w", encoding="utf-8") as f:
        for state, count in state_counts.items():
            f.write(f"{state}\t{count}\n")

    with open(transition_counts_file, "w", encoding="utf-8") as f:
        for (from_state, to_state), count in transition_counts.items():
            f.write(f"{from_state}\t{to_state}\t{count}\n")

    with open(emission_counts_file, "w", encoding="utf-8") as f:
        for (word, state), count in emission_counts.items():
            f.write(f"{word}\t{state}\t{count}\n")

    with open(all_states_file, "w", encoding="utf-8") as f:
        for state in all_states:
            f.write(f"{state}\n")

    with open(all_train_words_file, "w", encoding="utf-8") as f:
        for word in all_train_words:
            f.write(f"{word}\n")

    with open(all_test_words_file, "w", encoding="utf-8") as f:
        for word in all_test_words:
            f.write(f"{word}\n")

    with open(total_sentences_file, "w", encoding="utf-8") as f:
        f.write(f"{total_sentences}\n")

    print(f"Saved countabilities to {SAVE_DIR}")
# END save_counts


def load_counts():
    start_counts_file = f"{SAVE_DIR}/start_counts.tsv"
    state_counts_file = f"{SAVE_DIR}/state_counts.tsv"
    transition_counts_file = f"{SAVE_DIR}/transition_counts.tsv"
    emission_counts_file = f"{SAVE_DIR}/emission_counts.tsv"
    all_states_file = f"{SAVE_DIR}/all_states.txt"
    all_words_file = f"{SAVE_DIR}/all_words.txt"
    total_sentences_file = f"{SAVE_DIR}/total_sentences.txt"

    with open(start_counts_file, encoding="utf-8") as f:
        for line in f:
            state, count = line.strip().split("\t")
            start_counts[state] = int(count)

    with open(state_counts_file, encoding="utf-8") as f:
        for line in f:
            state, count = line.strip().split("\t")
            state_counts[state] = int(count)

    with open(transition_counts_file, encoding="utf-8") as f:
        for line in f:
            from_state, to_state, count = line.strip().split("\t")
            transition_counts[from_state, to_state] = int(count)

    with open(emission_counts_file, encoding="utf-8") as f:
        for line in f:
            word, state, count = line.strip().split("\t")
            emission_counts[word, state] = int(count)

    with open(all_states_file, encoding="utf-8") as f:
        for line in f:
            all_states.append(line.strip())

    with open(all_words_file, encoding="utf-8") as f:
        for line in f:
            all_train_words.append(line.strip())

    with open(total_sentences_file, encoding="utf-8") as f:
        global total_sentences
        total_sentences = int(f.readline().strip())

    print(f"Loaded countabilities from {SAVE_DIR}")
# END load_counts


def print_all():
    with open("out.txt", "w", encoding="utf-8") as f:
        f.write("Start Counts\n")
        for state, count in start_counts.items():
            f.write(f"{state}\t{count}\n")

        f.write("State Counts\n")
        for state, count in state_counts.items():
            f.write(f"{state}\t{count}\n")

        f.write("Transition Counts\n")
        for (from_state, to_state), count in transition_counts.items():
            f.write(f"{from_state}\t{to_state}\t{count}\n")

        f.write("Emission Counts\n")
        for (word, state), count in emission_counts.items():
            f.write(f"{word}\t{state}\t{count}\n")

        f.write("All States\n")
        for state in all_states:
            f.write(f"{state}\n")

        f.write("All Train Words\n")
        for word in all_train_words:
            f.write(f"{word}\n")

        f.write("All Test Words\n")
        for word in all_test_words:
            f.write(f"{word}\n")

        f.write(f"Total Sentences: {total_sentences}\n")
    # END with open("out.txt", "w", encoding="utf-8") as f
# END print_all


def main():
    train_loader = Loader(r"./data/train.txt")
    test_loader = Loader(r"./data/test.txt")
    calculate_all_counts(train_loader)
    save_counts()
    load_counts()
    print_all()
# END main


if __name__ == "__main__":
    main()
