from typing import List
import re


def preprocess_text(text: str):
    # Lowercasing
    text = text.lower()

    # Replace '\\' with space
    text = text.replace('\\', ' ')

    # Remove unnecessary symbols and non-alphanumeric characters
    # if a number like 123,456,789 is present, change it to 123456789
    text = re.sub(r'(\d),(\d)', r'\1\2', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

    return text
# END preprocess_text
