from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import json
import os

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
    return {'message': "Hi"}

@app.post("/send_frames/")
async def frames_to_model(test: Request):
    data = await test.json()
    print('successful data conversion')

    frames = np.array(json.loads(data))
    all_frames = [np.array(frame)for frame in frames]
    print(all_frames.shape)

    filename = 'frames.npz'

    if os.path.isfile(filename):
        with np.load(filename) as loaded_npz:
            loaded_npz['results'] = np.array(loaded_npz['results'].tolist() + all_frames)
    else:
        np.savez(filename, results = np.array(all_frames))

    return {"message": "Received data for prediction"
            , "prediction": "Hello I am mother (TEST DEFAULT)"}

@app.get("/predict/")
def prediction():
    # result = model.predict(np.array(full_frames))
    return {"prediction": "Hello I am mother (TEST DEFAULT)"}
