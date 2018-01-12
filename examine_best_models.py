import pickle
import os
import numpy as np
import pandas as pd
from lib.feature_importance import display_feature_importance


model_path = os.path.join('model','pca_logr')
model = pickle.load()

print(model.steps)
