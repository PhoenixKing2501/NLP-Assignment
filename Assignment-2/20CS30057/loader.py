from pathlib import Path
from typing import List


class Word(object):
    """
    Words description structure:

    <word_index> <word> <lemma> <pos_tag> <dependency_head_index> <dependency_label>

    """

    def __init__(
        self,
        record: str
    ):
        super(Word, self).__init__()

        self.record = record

        # Parse the record
        (word_index,
         self.word,
         self.norm_word,
         self.pos_tag,
         head_index,
         self.dependency_label) = record.split()

        self.word_index = int(word_index) - 1  # 0-based index
        # 0-based index [for root = -1]
        self.head_index = int(head_index) - 1

    # END __init__
    
    def copy(self):
        return Word(self.record)

# END class Sentence


class Sentence(object):
    """
    Basic structure of data in file:

    \\# sent_id \n
    <newline> \n
    \\# text \n
    <newline> \n
    \\# word descriptions (space separated, each word on a new line) \n
    <newline> \n
    """

    def __init__(
        self,
        sent_id: str,
        text: str,
        words: List[Word]
    ):
        super(Sentence, self).__init__()

        self.sent_id = sent_id
        self.text = text
        self.words = words
    # END __init__

    def __iter__(self):
        return iter(self.words)

    def __getitem__(self, index: int):
        return self.words[index]

    def __len__(self):
        return len(self.words)

    def copy(self):
        return Sentence(self.sent_id, self.text, [w.copy() for w in self.words])
    
# END class Sentence


class Loader(object):
    """
    Parse the data file and iterate over the Sentences.
    """

    def __init__(
        self,
        file_path: str
    ):
        self.file_path = file_path
        self.sentences: List[Sentence] = []
        self.__load_data()
    # END __init__

    def __load_data(self):
        with open(self.file_path, "r", encoding="utf-8") as file:
            sent_id = ""
            text = ""
            words: List[Word] = []

            for line in file:
                line = line.strip()

                if line.startswith("# sent_id"):
                    if sent_id:
                        self.sentences.append(Sentence(sent_id, text, words))
                    sent_id = line.split("=")[1].strip()
                    text = ""
                    words = []
                elif line.startswith("# text"):
                    text = line.split("=")[1].strip()
                elif line and line.split()[0].isdigit():
                    words.append(Word(line))
                # END if line.startswith
            # END for line in file

            if sent_id:
                self.sentences.append(Sentence(sent_id, text, words))
        # END with open
    # END load_data

    def __iter__(self):
        return iter(self.sentences)

    def __getitem__(self, index: int):
        return self.sentences[index]

    def __len__(self):
        return len(self.sentences)

# END class Loader


def main():
    loader = Loader("data/train.txt")

    sentence = loader[500]
    print(f'Sentence ID: {sentence.sent_id}')
    print(f'Text: {sentence.text}')

    print("Words:")
    for word in sentence:
        print(word)
        # break
    print()

    print(len(loader))
# END main


if __name__ == "__main__":
    main()
