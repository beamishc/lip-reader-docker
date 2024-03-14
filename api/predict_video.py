import numpy as np
import tensorflow as tf

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

def predict_video(model, video_frames):
    # Assuming `video_frames` is your input data # Ensure this is correctly implemented
    video_frames_batch = np.expand_dims(video_frames, axis=0)
    video_frames_batch = np.expand_dims(video_frames_batch, axis=-1)

    # Get predictions
    prediction = model.predict(video_frames_batch)

    # The sequence length should match the batch size of 'prediction'
    sequence_length = [len(video_frames)]  # Replace this with the correct sequence length for your data

    # Decode the predictions
    decoded_prediction = tf.keras.backend.ctc_decode(prediction, sequence_length, greedy=False)[0][0].numpy()
    predicted_text = tf.strings.reduce_join(num_to_char(decoded_prediction)).numpy().decode('utf-8')
    return predicted_text
