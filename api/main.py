from fastapi import Request, FastAPI
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from transformers import pipeline, RobertaTokenizerFast
from transformers import RobertaForMaskedLM

tokenizer = RobertaTokenizerFast.from_pretrained("srberta_tokenizer")
model = RobertaForMaskedLM.from_pretrained("./srberta_law_model")
model.to('cpu')

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
    fill = pipeline('fill-mask', model=model, tokenizer=tokenizer)
    fill(question.sentence)

    return fill
