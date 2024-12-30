import argparse
from pathlib import Path
import pickle
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

# Function to translate and save intermediate results
def process_pathology(pathology, output_dir, batch_size=100):
    output_dir = Path(output_dir)  # Ensure output_dir is a Path object
    output_dir.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist
    translations = []
    processed_indices = set()

    # Check for already processed files
    for file in output_dir.glob("translation_*.pkl"):
        idx = int(file.stem.split("_")[-1])  # Extract index from file name
        processed_indices.add(idx)

    for idx, item in pathology.iterrows():
        if idx in processed_indices:
            continue  # Skip already processed items

        try:
            translation = translate_text(item["presentedForm_data"])
        except Exception as e:
            print(f"Error processing index {idx}: {e}")
            continue

        translations.append({"index": idx, "translation": translation})

        # Save translation immediately as a .pkl file
        with (output_dir / f"translation_{idx}.pkl").open("wb") as f:
            pickle.dump({"index": idx, "translation": translation}, f)

        # Optional: Save after every batch to reduce memory usage
        if len(translations) >= batch_size:
            save_translations_to_csv(translations, output_dir)
            translations = []

    # Save remaining translations to CSV
    if translations:
        save_translations_to_csv(translations, output_dir)

# Function to save translations as CSV
def save_translations_to_csv(translations, output_dir):
    intermediate_csv = output_dir / "translations.csv"
    df = pd.DataFrame(translations)
    if intermediate_csv.exists():
        df_existing = pd.read_csv(intermediate_csv)
        df = pd.concat([df_existing, df], ignore_index=True)
    df.to_csv(intermediate_csv, index=False)

# Function to combine all translations into the final DataFrame
def combine_translations(pathology, output_dir):
    output_dir = Path(output_dir)  # Ensure output_dir is a Path object
    translations = []
    for file in output_dir.glob("translation_*.pkl"):
        print(file)
        with file.open("rb") as f:
            translations.append(pickle.load(f))

    # Combine into a single DataFrame
    df_translations = pd.DataFrame(translations).set_index("index")
    pathology = pathology.join(df_translations, how="left")
    return pathology

def run(data):
    data = Path(data)
    out = data.parent

    data = pd.read_csv(data)
    if "translation" in data.columns:
        raise ValueError(
            "It seems like to document already has translated reports... stopping here, please check"
        )

    # Note! Only doing pathology
    pathology = data[data["reportType"] == "Pathology"]
    pathology = pathology.dropna(subset=["presentedForm_data"])
    pathology = pathology.reset_index()

    process_pathology(pathology, output_dir=out / "pathology_translations")

    final_pathology = combine_translations(pathology, output_dir=out / "pathology_translations")
    final_pathology.to_csv("pathology_translated.csv", index=False)

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
