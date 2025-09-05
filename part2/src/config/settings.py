import dotenv
dotenv.load_dotenv(dotenv.find_dotenv(), override=True)
from pydantic_settings import BaseSettings, SettingsConfigDict
from sentence_transformers import SentenceTransformer
from typing import Optional
from ..config.logger import Logger

logger = Logger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8')
    POSTGRES_HOST: str
    POSTGRES_PORT: int
    POSTGRES_DB: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_ADMIN_USER: str
    POSTGRES_ADMIN_PASSWORD: str
    EMBEDDING_MODEL: str
    GOOGLE_API_KEY: str
    MAX_FILE_SIZE_MB: int
    BATCH_SIZE: int
    DEFAULT_QUERY_LIMIT: int
    COLLECTION: str

    _model: Optional[SentenceTransformer] = None

    @property
    def model(self):
        if self._model is None:
            self._model = SentenceTransformer(self.EMBEDDING_MODEL, device="cpu", trust_remote_code=True)
        return self._model

    @property
    def VECTOR_DIMENSION(self):
        return self.model.get_sentence_embedding_dimension()
    
try:
    settings = Settings(_env_file='.env', _env_file_encoding='utf-8')
except Exception as ex:
    logger.error(f"Could not get environmental variables: {ex}")