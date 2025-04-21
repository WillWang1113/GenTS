import src.model

model_names = src.model.__all__

for i in model_names:
    getattr(src.model, i)