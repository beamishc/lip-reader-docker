from initiate_model import initiate_model

def load_checkpoints():
    model=initiate_model()

    checkpoint_dir = 'model_mathilda_2000_12mar'

    epoch_number = 100  # for example, to load from checkpoint_epoch-06
    checkpoint_path = f"{checkpoint_dir}/checkpoint_epoch-{epoch_number:02d}"

    model.load_weights(checkpoint_path)

    return model
