import pickle


def get_model(models):
    return pickle.load(open(models, "rb"))


def model_pred(features, models):
    model = get_model(models)
    prediction = model.predict([features])
    return prediction
