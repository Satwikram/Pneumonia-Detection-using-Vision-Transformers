from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response

import tempfile
from pathlib import Path

import cv2

from transformers import ViTFeatureExtractor

from .apps import PredictConfig

# Create your views here.

img_size = (224, 224)
labels = ['pneumonia', 'normal']

def feature_extraction(samples):
  
  feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")

  return feature_extractor(samples, return_tensors="np")["pixel_values"]

class PredictionAPIView(APIView):

    def __init__(self) -> None:
        super().__init__()


    def post(self, request) -> Response:

        image = request.FILES['file']

        filename = str(image.name)

        temp = tempfile.TemporaryDirectory(
        prefix="pneumonia_")  

        temp_dir = Path(temp.name)
        
        # uFile = temp_dir / filename

        fn = f"{temp_dir}/test.png"

        with open(fn, "wb+") as f:
            f.write(image.read())
            
            img_array = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            resized_array = cv2.resize(img_array, img_size)
            resized_array = resized_array / 255

            pixel_values = feature_extraction(samples=[resized_array])

            pred = PredictConfig.model.predict(pixel_values)

            pred = 1 if pred >=0.5 else 0

            # print(pred)
            
            result = labels[pred]

        return Response({"Condition": result})