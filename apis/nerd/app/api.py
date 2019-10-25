from io import BytesIO
from urllib.parse import unquote

import torch
from fastapi import FastAPI
from torch import nn

from .aws import get_object_from_s3
from .nerd import NERD

model = NERD(backbone_dim=1024, alpha=0.8).eval()
model_state_dict = get_object_from_s3('model_state_dict.pt')
model.load_state_dict(torch.load(
    BytesIO(model_state_dict),
    map_location=torch.device('cpu')
))

app = FastAPI(
    title="Named Entity Recognition and Disambiguation",
    description="An neural entity linking system, capable of recognising entities in text and linking them to the wikipedia/wikidata knowledge base",
)


@app.get("/nerd")
def annotate(query_text: str):
    """
    annotate plain text with links to wikipedia pages
    """
    plain_text = unquote(query_text)
    response = model.annotate(plain_text)
    return {
        "request": query_text,
        "response": response
    }


@app.get('/nerd/health_check')
def health_check():
    return {'status': 'healthy'}
