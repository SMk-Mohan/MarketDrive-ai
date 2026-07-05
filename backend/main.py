from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from core.events import startup, shutdown
from api.routes import prediction, market, news, evaluation, report, health


@asynccontextmanager
async def lifespan(app: FastAPI):
    await startup()
    yield
    await shutdown()


app = FastAPI(
    title="MarketDrive AI",
    description="Multi-agent financial intelligence system for NSE stocks",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(prediction.router, prefix="/predict",  tags=["Prediction"])
app.include_router(market.router,     prefix="/market",   tags=["Market"])
app.include_router(news.router,       prefix="/news",     tags=["News"])
app.include_router(evaluation.router, prefix="/evaluate", tags=["Evaluation"])
app.include_router(report.router,     prefix="/report",   tags=["Report"])
app.include_router(health.router,                         tags=["Health"])