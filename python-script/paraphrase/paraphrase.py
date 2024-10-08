import os
import re
from pathlib import Path
import time
import nltk
import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from sentence_transformers import SentenceTransformer, util
import sacrebleu
from openai import OpenAI

nltk.download('punkt')
from nltk.tokenize import sent_tokenize

BASE_DIR = Path(__file__).resolve().parent

""""
Load a pre-trained BERT model. 'all-MiniLM-L6-v2' is a good balance between size and accuracy.
"""
model = SentenceTransformer('all-MiniLM-L6-v2')
client = OpenAI(api_key="KEYVALUE")


"""
Model used here is tuner007/pegasus_paraphrase, which is found to be better for paraphrase
"""
PEGASUS_MODEL_NAME: str = 'tuner007/pegasus_paraphrase'
PEGASUS_MODEL_PATH: str = os.path.join(BASE_DIR, 'pegasus_paraphrase_model/Trials/models/pegasus')
PEGASUS_TOKENIZER_PATH: str = os.path.join(BASE_DIR, 'pegasus_paraphrase_tokenize/Trials/models/pegasus')


class PegasusParaphraseGenerator:
    def __init__(self):
        self.model_name = PEGASUS_MODEL_NAME
        self.model_path = PEGASUS_MODEL_PATH
        self.tokenizer_path = PEGASUS_TOKENIZER_PATH
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.load_model()

    def load_model(self):
        # Check if the model and tokenizer are already downloaded
        if not os.path.exists(self.model_path):
            print(f"Downloading and saving the model to {self.model_path}...")
            model = PegasusForConditionalGeneration.from_pretrained(self.model_name)
            model.save_pretrained(self.model_path)
        else:
            model = PegasusForConditionalGeneration.from_pretrained(self.model_path)

        if not os.path.exists(self.tokenizer_path):
            print(f"Downloading and saving the tokenizer to {self.tokenizer_path}...")
            tokenizer = PegasusTokenizer.from_pretrained(self.model_name)
            tokenizer.save_pretrained(self.tokenizer_path)
        else:
            tokenizer = PegasusTokenizer.from_pretrained(self.tokenizer_path)

        self.model = model.to(self.device)
        self.tokenizer = tokenizer

    def get_response(self, input_text, num_return_sequences, num_beams=10):
        batch = self.tokenizer([input_text], truncation=True, padding='longest', max_length=60, return_tensors="pt").to(
            self.device)
        translated = self.model.generate(**batch, max_length=60, num_beams=num_beams,
                                         num_return_sequences=num_return_sequences, temperature=1.5)
        tgt_text = self.tokenizer.batch_decode(translated, skip_special_tokens=True)
        return tgt_text


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


def main():
    """

    Check if input paragraph has 200-400 words, else it would return asking to enter required sized text
    :return:
    """
    input_text = input("Enter a paragraph (200-400 words): ")
    is_valid, cleaned_text, word_count_ = validate_input(input_text)
    min_word_count = word_count_ * 0.8
    if not is_valid:
        print("Input text must be between 200 and 400 words.")
        return "Input text must be between 200 and 400 words."


    """
    Load pretrained model. Here we are returning 5 outputs.
    This is to ensure that, if a paraphrase output has less than 80% of input, then we reject it and goes to next
    """
    paraphrase_generator = PegasusParaphraseGenerator()

    # Measure latency and generate paraphrase with pegasus
    start_time = time.time()
    sentences = sent_tokenize(cleaned_text)
    paraphrase_1 = []
    paraphrase_2 = []
    paraphrase_3 = []
    paraphrase_4 = []
    paraphrase_5 = []
    for sentence in sentences:
        res = paraphrase_generator.get_response(sentence, 5)
        paraphrase_1.append(res[0])
        paraphrase_2.append(res[1])
        paraphrase_3.append(res[2])
        paraphrase_4.append(res[3])
        paraphrase_5.append(res[4])
    cpg_outs = [' '.join(paraphrase_1), ' '.join(paraphrase_2), ' '.join(paraphrase_3), ' '.join(paraphrase_4),
                ' '.join(paraphrase_5)]

    cpg_outs = [x for x in cpg_outs if get_word_count(x) >= min_word_count]
    cpg_latency = time.time() - start_time

    # Measure latency and generate paraphrase with GPT-3
    start_time = time.time()
    gpt3_paraphrase = paraphrase_with_openai(cleaned_text)
    gpt3_latency = time.time() - start_time

    # For qualitative measurement between LLM output and cpg output
    bleu_scores = []
    for cpg_out in cpg_outs:
        bleu_score = calculate_bleu_score(gpt3_paraphrase, cpg_out)
        bleu_scores.append(bleu_score)

    bert_similarity_scores = []
    for cpg_out in cpg_outs:
        similarity_scores = semantic_similarity(gpt3_paraphrase, cpg_out)
        bert_similarity_scores.append(similarity_scores)
    print({'cpg_output': cpg_outs, 'cpg_latency': cpg_latency, 'openai_paraphrase': gpt3_paraphrase,
           'openai_latency': gpt3_latency, 'bleu_scores': bleu_scores, 'similarity_scores': bert_similarity_scores})
    return {'cpg_output': cpg_outs, 'cpg_latency': cpg_latency, 'openai_paraphrase': gpt3_paraphrase,
            'openai_latency': gpt3_latency, 'bleu_scores': bleu_scores, 'similarity_scores': bert_similarity_scores}


if __name__ == "__main__":
    main()
