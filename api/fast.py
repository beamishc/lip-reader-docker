from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from api.load_checkpoints import load_checkpoints
from api.predict_video import predict_video
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
    print(len(all_frames))
    print(all_frames[0].shape)

    filename = 'frames.npz'

    if os.path.isfile(filename):
        with np.load(filename) as loaded_npz:
            np.savez(filename, results = np.array(loaded_npz['results'].tolist() + all_frames))
    else:
        np.savez(filename, results = np.array(all_frames))

    return {"message": "Received data for prediction"}

@app.get("/predict/")
def prediction():
    with np.load('frames.npz') as loaded_npz:
            full_frames = loaded_npz['results']
    final_form = full_frames.shape
    if final_form[0] < 75:
        return {'prediction': 'not enough frames for prediction'
            , "num_of_frames_provided": str(final_form)}
    frames_list = full_frames.tolist()
    splits = []
    while len(frames_list) >= 75:
        splits.append(frames_list[:75])
        frames_list = frames_list[75:]
    model = load_checkpoints()
    result = ''
    for x in splits:
        if x.shape[0] == 75:
            result = result + predict_video(model, x)
    return {"prediction": result
            , "num_of_frames_provided": str(final_form)}
