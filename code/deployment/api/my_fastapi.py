import sys
sys.path.append('../../models')

from my_model import MyModel
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import numpy as np  
import ast

app = FastAPI()
model = MyModel()
model.load_state_dict(torch.load('../../../models/baseline', weights_only=True))
model.eval()

class InputImage(BaseModel):
    image: str

@app.post("/predict")
def predict(input: InputImage):
    
    image = np.array(ast.literal_eval(input.image))
    return {"prediction": model(torch.tensor(image, dtype=torch.float32).view(1, 1, 28, 28)).argmax(-1).item()}