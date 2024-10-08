import os
from pathlib import Path
# from pydantic import BaseSettings
from pydantic import AnyHttpUrl
from pydantic_settings import BaseSettings
from typing import List

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    API_STR: str = '/api'
    # SERVER_NAME: str
    # SERVER_HOST: AnyHttpUrl
    """
    BACKEND_CORS_ORIGINS is a JSON-formatted list of origins
    e.g: '["http://localhost", "http://localhost:4200", "http://localhost:3000", \
    "http://localhost:8080", "http://local.dockertoolbox.tiangolo.com"]
    """
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = []
    BERT_BATCH_SIZE: int = 4
    MIN_WORDS: int = 4
    MAX_WORDS: int = 200
    MODEL_NAME: str = "Vamsi/T5_Paraphrase_Paws"
    MODEL_PATH: str = os.path.join(BASE_DIR, 'paraphrase_model/Trials/models/Vamsi_T5_Paraphrase_Paws')
    TOKENIZER_PATH: str = os.path.join(BASE_DIR, 'paraphrase_tokenizer/Trials/models/Vamsi_T5_Paraphrase_Paws')

    PEGASUS_MODEL_NAME: str = 'tuner007/pegasus_paraphrase'
    PEGASUS_MODEL_PATH: str = os.path.join(BASE_DIR, 'pegasus_paraphrase_model/Trials/models/pegasus')
    PEGASUS_TOKENIZER_PATH: str = os.path.join(BASE_DIR, 'pegasus_paraphrase_tokenize/Trials/models/pegasus')


settings = Settings()
