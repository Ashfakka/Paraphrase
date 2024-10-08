from fastapi import APIRouter
import nltk
import time
from core.model_config import pegasus_paraphrase_model  # ,paraphrase_model
from pydantic import BaseModel

nltk.download('punkt')
from nltk.tokenize import sent_tokenize

from api.utils import validate_input, paraphrase_with_openai, get_word_count, calculate_bleu_score, semantic_similarity

router = APIRouter()


class ParaphraseInput(BaseModel):
    paragraph: str


@router.put('/generate/')
async def paraphrase_generator(input_data: ParaphraseInput):
    is_valid, cleaned_text, word_count_ = validate_input(input_data.paragraph)
    min_word_count = word_count_ * 0.8
    if not is_valid:
        return 'Input text must be between 200 and 400 words.'

    # Measure latency and generate paraphrase with pegasus
    start_time = time.time()
    sentences = sent_tokenize(cleaned_text)
    paraphrase_1 = []
    paraphrase_2 = []
    paraphrase_3 = []
    paraphrase_4 = []
    paraphrase_5 = []
    for sentence in sentences:
        res = pegasus_paraphrase_model.get_response(sentence, 5)
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

    return {'cpg_output': cpg_outs, 'cpg_latency': cpg_latency, 'openai_paraphrase': gpt3_paraphrase,
            'openai_latency': gpt3_latency, 'bleu_scores': bleu_scores, 'similarity_scores': bert_similarity_scores}
