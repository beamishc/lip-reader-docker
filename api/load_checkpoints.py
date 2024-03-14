from api.initiate_model import initiate_model

def load_checkpoints():
    model=initiate_model()

    # checkpoint_dir = 'api/final_model'

    epoch_number = 100  # for example, to load from checkpoint_epoch-06
    checkpoint_path = f"api/final_model/checkpoint_epoch-{epoch_number:02d}"

    model.load_weights(checkpoint_path)

    return model
