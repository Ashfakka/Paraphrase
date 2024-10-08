from fastapi import FastAPI, APIRouter
from api.api import api_router
from core.config import settings
from starlette.middleware.cors import CORSMiddleware
# from core.model_config import paraphrase_model

app = FastAPI()
# paraphrase_model

# Set all CORS enabled origins
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(api_router, prefix=settings.API_STR)
