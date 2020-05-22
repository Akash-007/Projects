import pandas
import tensorflow.keras.models as model

model_new = model.load_model('multiclass_model.h5')

query = 'Feds Plosser: Taper pace may be too slow'
res = model_new.predict_classes(query.lower())
print(res)