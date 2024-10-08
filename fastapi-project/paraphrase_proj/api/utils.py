import re
from openai import OpenAI
import sacrebleu
from sentence_transformers import SentenceTransformer, util
import torch

# Load a pre-trained BERT model. 'all-MiniLM-L6-v2' is a good balance between size and accuracy.
model = SentenceTransformer('all-MiniLM-L6-v2')
client = OpenAI(api_key="OPENAI KEY")


def preprocess_text(text):
    # Remove all special characters except for essential punctuation, replace them with spaces
    text = re.sub(r'[^a-zA-Z0-9.,!?\'\s]', ' ', text)

    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def get_word_count(processed_text):
    return len(processed_text.split())


def validate_input(text):
    # Preprocess the text first
    processed_text = preprocess_text(text)

    # Count words in the cleaned, processed text
    word_count = get_word_count(processed_text)
    return 200 <= word_count <= 400, processed_text, word_count


def paraphrase_with_openai(text):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are an intelligent paraphrase generator. \nYour objectives are,\n1.  Paraphrase given input\n2. Your output should have a minimum length of 80% of the input text length."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Paraphrase the following: \n{text}"
                    }
                ]
            }
        ],
        temperature=1,
        max_tokens=1024,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    openai_response = response.choices[0].message.content
    return openai_response


def calculate_bleu_score(original, paraphrased):
    bleu_score = sacrebleu.corpus_bleu([paraphrased], [[original]]).score
    return bleu_score

def semantic_similarity(paragraph1, paragraph2):
    # Encode the paragraphs to get their embeddings
    embedding1 = model.encode(paragraph1, convert_to_tensor=True)
    embedding2 = model.encode(paragraph2, convert_to_tensor=True)

    # Compute cosine similarity
    cosine_similarity = util.pytorch_cos_sim(embedding1, embedding2)

    # Return the cosine similarity score as a Python float
    return cosine_similarity.item()