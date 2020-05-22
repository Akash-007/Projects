from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
raw_data = pd.read_csv(r'C:\Users\HP\AI Projects\Learn\Kaggle\m_nist\test\test.csv')
X = np.array(raw_data)
X = X.reshape(X.shape[0], 1, 28, 28, 1)
X = X/255
model_ = load_model('new_model.h5')
print(X[0].shape)
print(model_.predict_classes(X[0]))
plt.imshow(X[0].reshape(28, 28))
plt.show()
'''
for i in range(20):
    print(model_.predict_classes(X[i:i+1]))
    plt.imshow(X[i].reshape(28, 28))
    plt.show()

#print(y_pred)
#result = pd.DataFrame(y_pred)
#result.to_csv('submiss_new.csv')
'''
