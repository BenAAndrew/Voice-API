import re
from unidecode import unidecode

COMMA_NUMBER_RE = re.compile(r"([0-9][0-9\,]+[0-9])")
DECIMAL_NUMBER_RE = re.compile(r"([0-9]+\.[0-9]+)")
NUMBER_RE = re.compile(r"[0-9]+")
ORDINALS = re.compile(r"([0-9]+[st|nd|rd|th]+)")
CURRENCY = re.compile(r"([£|$|€]+[0-9]+)")
WHITESPACE_RE = re.compile(r"\s+")
ALLOWED_CHARACTERS_RE = re.compile("[^a-z ,.!?'-]+")

MONETARY_REPLACEMENT = {"$": " dollars", "£": " pounds", "€": " euros"}
ABBREVIATION_REPLACEMENT = {
    "mr.": "mister",
    "mrs.": "misess",
    "dr.": "doctor",
    "no.": "number",
    "st.": "saint",
    "co.": "company",
    "jr.": "junior",
    "maj.": "major",
    "gen.": "general",
    "drs.": "doctors",
    "rev.": "reverend",
    "lt.": "lieutenant",
    "hon.": "honorable",
    "sgt.": "sergeant",
    "capt.": "captain",
    "esq.": "esquire",
    "ltd.": "limited",
    "col.": "colonel",
    "ft.": "fort",
}


def clean_text(text, inflect_engine):
    text = unidecode(text)
    text = text.strip()
    text = text.lower()
    # Convert currency to words
    money = re.findall(CURRENCY, text)
    for amount in money:
        for key, value in MONETARY_REPLACEMENT.items():
            if key in amount:
                text = text.replace(amount, amount[1:] + value)
    # Convert ordinals to words
    ordinals = re.findall(ORDINALS, text)
    for ordinal in ordinals:
        text = text.replace(ordinal, inflect_engine.number_to_words(ordinal))
    # Convert comma & decimal numbers to words
    numbers = re.findall(COMMA_NUMBER_RE, text) + re.findall(DECIMAL_NUMBER_RE, text)
    for number in numbers:
        text = text.replace(number, inflect_engine.number_to_words(number))
    # Convert standard numbers to words
    numbers = re.findall(NUMBER_RE, text)
    for number in numbers:
        text = text.replace(number, inflect_engine.number_to_words(number))
    # Replace abbreviations
    for key, value in ABBREVIATION_REPLACEMENT.items():
        text = text.replace(" " + key + " ", " " + value + " ")
    # Collapse whitespace
    text = re.sub(WHITESPACE_RE, " ", text)
    # Remove banned characters
    text = re.sub(ALLOWED_CHARACTERS_RE, "", text)
    return text
