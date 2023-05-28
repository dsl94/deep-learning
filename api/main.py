from fastapi import Request, FastAPI
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from guess import guess

class Question(BaseModel):
    sentence: str

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/guess")
async def search(question: Question):
    return guess.guess(question.sentence)
