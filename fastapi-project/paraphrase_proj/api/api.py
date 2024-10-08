from fastapi import APIRouter
from paraphrase.jobs import routers as paraphrase_router

api_router = APIRouter()

api_router.include_router(paraphrase_router.router, prefix='/paraphrase', tags=['jobs'])
