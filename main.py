from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import boto3
import os
from botocore.exceptions import NoCredentialsError, ClientError

app = FastAPI(title="AWS Rekognition API", version="1.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize AWS Rekognition client with error handling
rekognition = None
try:
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    
    if not aws_access_key or not aws_secret_key:
        print("Warning: AWS credentials not found in environment variables")
    else:
        rekognition = boto3.client(
            "rekognition",
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name="us-east-1",
        )
        print("AWS Rekognition client initialized successfully")
        
except NoCredentialsError:
    print("Error: AWS credentials not found")
except Exception as e:
    print(f"Error initializing AWS client: {str(e)}")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "FastAPI AWS Rekognition API is running",
        "status": "healthy",
        "aws_configured": rekognition is not None
    }

@app.get("/health")
async def health_check():
    """Detailed health check with AWS status"""
    aws_status = "configured" if rekognition else "not configured"
    return {
        "status": "healthy",
        "aws_rekognition": aws_status,
        "environment_vars": {
            "AWS_ACCESS_KEY_ID": bool(os.getenv("AWS_ACCESS_KEY_ID")),
            "AWS_SECRET_ACCESS_KEY": bool(os.getenv("AWS_SECRET_ACCESS_KEY"))
        }
    }

@app.post("/detect")
async def detect_labels(file: UploadFile = File(...)):
    """Detect labels in uploaded image using AWS Rekognition"""
    
    # Check if AWS client is initialized
    if not rekognition:
        raise HTTPException(
            status_code=500, 
            detail="AWS Rekognition client not configured. Please check environment variables."
        )
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail="File must be an image (JPEG, PNG, etc.)"
        )
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Check file size (AWS Rekognition has limits)
        if len(image_bytes) > 5 * 1024 * 1024:  # 5MB limit
            raise HTTPException(
                status_code=400, 
                detail="Image file too large. Maximum size is 5MB."
            )
        
        if len(image_bytes) == 0:
            raise HTTPException(
                status_code=400, 
                detail="Empty file uploaded"
            )
        
        # Call AWS Rekognition
        response = rekognition.detect_labels(
            Image={'Bytes': image_bytes},
            MaxLabels=10,
            MinConfidence=75
        )
        
        # Process and format results
        labels = [
            {
                "name": label["Name"], 
                "confidence": round(label["Confidence"], 2),
                "categories": [category["Name"] for category in label.get("Categories", [])]
            }
            for label in response["Labels"]
        ]
        
        return {
            "success": True,
            "filename": file.filename,
            "labels_count": len(labels),
            "labels": labels
        }
    
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        raise HTTPException(
            status_code=400, 
            detail=f"AWS Rekognition Error [{error_code}]: {error_message}"
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/debug")
async def debug_environment():
    """Debug endpoint to check environment configuration"""
    return {
        "environment_variables": {
            "AWS_ACCESS_KEY_ID": "Set" if os.getenv("AWS_ACCESS_KEY_ID") else "Not Set",
            "AWS_SECRET_ACCESS_KEY": "Set" if os.getenv("AWS_SECRET_ACCESS_KEY") else "Not Set"
        },
        "aws_client_status": "Initialized" if rekognition else "Not Initialized",
        "region": "us-east-1"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
