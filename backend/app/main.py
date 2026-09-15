from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.ml.model_loader import load_model_bundle
from app.database import Base, engine
from app.scheduler import start_scheduler
from app.routers import transactions, predictions, feedback, monitoring, model as model_router

app = FastAPI(title="DriftGuard API")

# The frontend (Vite dev server, typically localhost:5173) runs on a
# different origin than the API (localhost:8000) — browsers block
# cross-origin requests by default unless the server explicitly allows
# it. This is a browser security feature, not a bug to work around
# quietly; it's worth being able to explain in an interview.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    Base.metadata.create_all(bind=engine)
    load_model_bundle()
    app.state.scheduler = start_scheduler()


@app.on_event("shutdown")
def shutdown_event():
    scheduler = getattr(app.state, "scheduler", None)
    if scheduler is not None:
        scheduler.shutdown(wait=False)


app.include_router(transactions.router, tags=["transactions"])
app.include_router(predictions.router, tags=["predictions"])
app.include_router(feedback.router, tags=["feedback"])
app.include_router(monitoring.router, tags=["monitoring"])
app.include_router(model_router.router, tags=["model"])


@app.get("/health")
def health():
    return {"status": "ok"}
