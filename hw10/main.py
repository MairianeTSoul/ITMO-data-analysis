import joblib
import uvicorn
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse

app = FastAPI()

with open("../real_estate_model.pkl", 'rb') as file:
    model, columns = joblib.load(file)


class ModelRequestData(BaseModel):
    total_square: float
    rooms: int
    floor: int

class Result(BaseModel):
    result: float

@app.get("/health")
def health():
    return JSONResponse(content={"message": "It's alive!"}, status_code=200)

@app.get("/predict_get")
def predict_get(total_square: float, rooms: int, floor: int):
    input_df = pd.DataFrame([[total_square, rooms, floor]], columns=columns)
    result = model.predict(input_df)[0]
    return Result(result=result)

@app.post("/predict_post", response_model=Result)
def predict_post(data: ModelRequestData):
    input_data = data.dict()
    input_df = pd.DataFrame(input_data, index=[0])
    result = model.predict(input_df)[0]
    return Result(result=result)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
