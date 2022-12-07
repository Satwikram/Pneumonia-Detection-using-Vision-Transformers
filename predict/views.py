from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response

import tempfile
from pathlib import Path

import cv2

# Create your views here.

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
            
            image = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)

    

        return Response({"Success"})