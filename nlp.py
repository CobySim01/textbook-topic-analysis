import spacy


nlp = spacy.load("en_core_web_sm")


def lemmatize(sentence):
    return [
        token.lemma_ for token in nlp(sentence) if not token.is_stop and token.is_alpha
    ]
