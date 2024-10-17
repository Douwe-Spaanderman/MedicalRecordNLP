import argparse
from pathlib import Path
import os
import pandas as pd

from transformers import MarianMTModel, MarianTokenizer

# Globally setting this pretrained pretrained transformer model
model_name = "Helsinki-NLP/opus-mt-nl-en"
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)


def translate_sentence(sentence: str) -> str:
    """
    Translates a single sentence using a pre-trained translation model.

    Parameters
    ----------
    sentence : str
        The input sentence to be translated.

    Returns
    -------
    str
        The translated sentence.
    """
    input_ids = tokenizer.encode(sentence, return_tensors="pt")
    translation_ids = model.generate(input_ids, num_return_sequences=1)
    translated_sentence = tokenizer.decode(translation_ids[0], skip_special_tokens=True)
    return translated_sentence


def translate_text(text: str) -> str:
    """
    Translates multiple paragraphs of text while preserving the file structure.

    Handles long sentences and keeps paragraph and sentence breaks in the original format.
    If a sentence exceeds the token limit, it splits the sentence into smaller parts
    for translation, and very long mutation lists are not translated.

    Parameters
    ----------
    text : str
        The input text containing multiple paragraphs and sentences to be translated.

    Returns
    -------
    str
        The translated text with the same paragraph and sentence structure as the original.
    """
    paragraphs = text.split("\r\n\r\n")

    translated_text = ""
    for paragraph in paragraphs:
        # This is done to keep the file structure
        if not paragraph:
            translated_text += "\r\n\r\n"
            continue

        sentences = paragraph.split("\r\n")
        for sentence in sentences:
            # This is done to keep the file structure
            if not sentence:
                translated_text += "\r\n"
                continue

            if len(sentence) > 512:
                words = sentence.split(". ")
                for w in words:
                    if not w:
                        translated_text += ". "
                        continue

                    if len(w) > 512:
                        # These are actually just long mutation list and don't need translation
                        translated_text += w + ". "
                    else:
                        translated_w = translate_sentence(w)
                        translated_text += translated_w + ". "
            else:
                translated_sentence = translate_sentence(sentence)
                translated_text += translated_sentence + "\r\n"

        # This is done to keep the file structure
        translated_text += "\r\n\r\n"

    return translated_text


def run(data):
    data = Path(data)
    out = data.parent / "pathology_translated.csv"

    data = pd.read_csv(data)
    if "Translation" in data.columns:
        raise ValueError(
            "It seems like to document already has translated reports... stopping here, please check"
        )

    # Note! Only doing pathology
    pathology = data[data["reportType"] == "Pathology"]
    pathology = pathology.dropna(subset=["presentedForm_data"])
    pathology = pathology.reset_index()

    translations = []
    for idx, item in pathology.iterrows():
        print(idx)
        translation = translate_text(item["presentedForm_data"])
        translations.append(translation)

    pathology = pathology
    pathology["Translation"] = translations

    pathology.to_csv(out)


def main():
    parser = argparse.ArgumentParser(
        description="Dutch - English translation of csv export"
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=str,
        help="Path to the dutch 'verslagen' document",
    )

    args = parser.parse_args()

    run(args.input)


if __name__ == "__main__":
    main()
