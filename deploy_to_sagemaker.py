import boto3
import json
import time

# Initialize the SageMaker client
region = 'us-east-1'  # Replace with your AWS region
sagemaker_client = boto3.client('sagemaker', region_name=region)

# Variables
ecr_image_uri = '021891598302.dkr.ecr.us-east-1.amazonaws.com/rental-prediction-ecr:latest'
sagemaker_role_arn = 'arn:aws:iam::021891598302:role/service-role/SageMaker-mlops-test-user-295'  # Replace with your IAM Role ARN
model_name = 'rental-prediction-model'
endpoint_config_name = 'rental-prediction-endpoint-config'
endpoint_name = 'rental-prediction-endpoint'

# Step 1: Create a SageMaker model
print("Creating SageMaker model...")
response = sagemaker_client.create_model(
    ModelName=model_name,
    PrimaryContainer={
        'Image': ecr_image_uri,
        'Mode': 'SingleModel',
    },
    ExecutionRoleArn=sagemaker_role_arn
)
print("Model created:", response['ModelArn'])

# Step 2: Create an endpoint configuration
print("Creating endpoint configuration...")
response = sagemaker_client.create_endpoint_config(
    EndpointConfigName=endpoint_config_name,
    ProductionVariants=[
        {
            'VariantName': 'AllTraffic',
            'ModelName': model_name,
            'InitialInstanceCount': 1,
            'InstanceType': 'ml.m5.large'  # Choose the instance type based on your needs
        },
    ]
)
print("Endpoint configuration created:", response['EndpointConfigArn'])

# Step 3: Deploy the SageMaker endpoint
print("Deploying the endpoint...")
response = sagemaker_client.create_endpoint(
    EndpointName=endpoint_name,
    EndpointConfigName=endpoint_config_name
)
print("Endpoint creation initiated:", response['EndpointArn'])

# Step 4: Wait for the endpoint to be in service
print("Waiting for endpoint to be in service...")
while True:
    response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
    status = response['EndpointStatus']
    print("Endpoint Status:", status)

    if status == 'InService':
        print("Endpoint is ready!")
        break
    elif status == 'Failed':
        raise Exception("Endpoint deployment failed!")
    
    time.sleep(60)  # Wait for 1 minute before checking the status again

# Get the Endpoint URL
endpoint_url = f"https://runtime.sagemaker.{region}.amazonaws.com/endpoints/{endpoint_name}/invocations"
print("Endpoint URL:", endpoint_url)

print(f"You can test the endpoint using the URL: {endpoint_url}")
