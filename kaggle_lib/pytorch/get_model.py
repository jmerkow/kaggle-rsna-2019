import models_pytorch.classification as mpc


def get_model(base=None, **model_params):
    # only 1 library exists, will update later
    return mpc.get_model(**model_params)
