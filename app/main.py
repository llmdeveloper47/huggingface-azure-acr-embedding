import torch
from fastapi import FastAPI, Response
from fastapi.responses import PlainTextResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from typing import List
from app.model.model import generate_embeddings


app = FastAPI()

class TextIn(BaseModel):
    text: List[str]


class PredictionOut(BaseModel):
    embeddings: List[List[float]]


@app.get('/')
def root():
    return Response("<h1>An embedding API to generate embeddings of text documents</h1>")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return PlainTextResponse(str(exc), status_code=400)

@app.post('/predict')
def predict(payload: TextIn):
    
    print(payload.text)
    sentence_embeddings = generate_embeddings(payload.text)
    print(f"sentence_embeddings in main.py: {sentence_embeddings}")

    return PredictionOut(embeddings = sentence_embeddings)
