from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import boto3
import os


app = FastAPI()

# CORS (change to allow only your frontend URL in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rekognition = boto3.client(
    "rekognition",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name="us-east-1",
)

@app.post("/detect")
async def detect_labels(file: UploadFile = File(...)):
    image_bytes = await file.read()

    response = rekognition.detect_labels(
        Image={'Bytes': image_bytes},
        MaxLabels=10,
        MinConfidence=75
    )

    labels = [
        {"name": label["Name"], "confidence": round(label["Confidence"], 2)}
        for label in response["Labels"]
    ]
    return {"labels": labels}
