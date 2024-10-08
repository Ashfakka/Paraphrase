from transformers import AutoTokenizer, AutoModel, pipeline, AutoModelForSeq2SeqLM, PegasusForConditionalGeneration, PegasusTokenizer
import torch
import os

from core.config import settings


class ParaphraseModel:
    def __init__(self, model_name, device=-1, small_memory=True, batch_size=settings.BERT_BATCH_SIZE):
        self.model_name = model_name
        self._set_device(device)
        self.small_device = 'cpu' if small_memory else self.device
        self.batch_size = batch_size
        self.load_pretrained_model()

    def _set_device(self, device):
        if device == -1 or device == 'cpu':
            self.device = 'cpu'
        elif device == 'cuda' or device == 'gpu':
            self.device = 'cuda'
        elif isinstance(device, int) or isinstance(device, float):
            self.device = 'cuda'
        else:  # default
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")

    def load_pretrained_model(self):
        # Check if the model and tokenizer have been downloaded; if not, download them
        if not os.path.exists(settings.MODEL_PATH):
            print(f"Downloading model to {settings.MODEL_PATH}")
            os.makedirs(settings.MODEL_PATH, exist_ok=True)
            model = AutoModelForSeq2SeqLM.from_pretrained(settings.MODEL_NAME)
            model.save_pretrained(settings.MODEL_PATH)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(settings.MODEL_PATH)

        if not os.path.exists(settings.TOKENIZER_PATH):
            print(f"Downloading tokenizer to {settings.TOKENIZER_PATH}")
            os.makedirs(settings.TOKENIZER_PATH, exist_ok=True)
            tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_NAME)
            tokenizer.save_pretrained(settings.TOKENIZER_PATH)
        else:
            tokenizer = AutoTokenizer.from_pretrained(settings.TOKENIZER_PATH)

        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.pipeline = pipeline('text-generation', model=self.model, tokenizer=self.tokenizer,
                                 device=0 if self.device == 'cuda' else -1)

    def paraphrase(self, text):
        return self.pipeline(text, max_length=512, num_return_sequences=1)


class PegasusParaphraseGenerator:
    def __init__(self):
        self.model_name = settings.PEGASUS_MODEL_NAME
        self.model_path = settings.PEGASUS_MODEL_PATH
        self.tokenizer_path = settings.PEGASUS_TOKENIZER_PATH
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
        batch = self.tokenizer([input_text], truncation=True, padding='longest', max_length=60, return_tensors="pt").to(self.device)
        translated = self.model.generate(**batch, max_length=60, num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=1.5)
        tgt_text = self.tokenizer.batch_decode(translated, skip_special_tokens=True)
        return tgt_text