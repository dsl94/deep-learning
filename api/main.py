from fastapi import Request, FastAPI
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from transformers import pipeline, RobertaTokenizerFast
from transformers import RobertaForMaskedLM
import uvicorn

from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("./SRBerta")

model = AutoModelForMaskedLM.from_pretrained("./SRBerta")
model.to('cpu')
model.eval()

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
    a = fill(question.sentence)

    return a

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
