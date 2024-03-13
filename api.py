from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import json

app = FastAPI()

origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

@app.get("/home/")
def home():
    return {'message': "Hello"}

@app.post("/predict/")
async def numpy_test(test: Request):
    data = await test.json()
    print('successful data conversion')
    first_frame = np.array(np.array(json.loads(data))[0])

    frames = np.array(json.loads(data))
    all_frames = np.array([np.array(frame)for frame in frames])
    print(all_frames.shape)
    return {"message": "Received data for prediction"
            , "data": json.dumps(first_frame.tolist())
            , "prediction": "Hello I am mother (TEST DEFAULT)"}
