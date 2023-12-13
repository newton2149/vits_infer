""" from https://github.com/keithito/tacotron """
from text import cleaners
from text.symbols import symbols

from gruut import sentences
from cvutils import Phonemiser

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}


def kinyarwanda(text):
    phonemes= []
    p = Phonemiser('rw')
    text_split = text.split(" ")
    for t in text_split:
        phonemes.append(p.phonemise(t))
    phonemes = " ".join(phonemes)
    print(f'Text: {text} ---> Phoneme: {phonemes} ',end='\r')
    return phonemes


def phonemize_gruut(text):


    phoneme_sentence = []

    for sent in sentences(text, lang="fr-fr"):
        for word in sent:
            if word.phonemes:
                phoneme_sentence.extend(word.phonemes)
            phoneme_sentence.extend(" ")


    phoneme = "".join(phoneme_sentence).strip()
    phoneme = phoneme.replace("|","*")


    return phoneme

def text_to_sequence(text, cleaner_names):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through
    Returns:
      List of integers corresponding to the symbols in the text
    """
    sequence = []

    # clean_text = _clean_text(text, cleaner_names)

    clean_text = _clean_text(text, cleaner_names)


    for symbol in clean_text:
        symbol_id = _symbol_to_id[symbol]
        sequence += [symbol_id]
    return sequence


def cleaned_text_to_sequence(cleaned_text):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
    """
    sequence = [_symbol_to_id[symbol] for symbol in cleaned_text]
    return sequence


def sequence_to_text(sequence):
    """Converts a sequence of IDs back to a string"""
    result = ""
    for symbol_id in sequence:
        s = _id_to_symbol[symbol_id]
        result += s
    return result


def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise Exception("Unknown cleaner: %s" % name)
        text = cleaner(text)
    return text
