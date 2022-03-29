#! /usr/bin/env python
# -*- coding: utf-8 -*-

import json
import logging
import subprocess
import re
import sys
import numpy as np
import unicodedata
import traceback
from tensorflow import keras

def handler(event, context):
    models = {
        "phone_spam": "/app/phone_spam.h5", 
        "card_fraud": "/app/card_fraud.hp",
    }

    try:
        data = json.loads(event['body'])

        if 'instances' not in data:
            return "instances no set"

	if 'model' not in data:
	    return "model not set"
	    
	modelName = data["model"]

	if modelName not in models:
	    return "incorrect_model"
	    
	modelPath = models[modelName]

        model = keras.models.load_model(modelPath)
        npArray = np.array(data["instances"])
        predictions = model.predict(npArray)     
        print(predictions)
        
        return json.dumps([{"prob": predictions.tolist()}])
    except Exception as e:
        return "erorr: "+str(e)+traceback.format_exc()+"\nEvent: "+str(event)
