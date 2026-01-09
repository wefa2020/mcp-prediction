#!/bin/bash
set -e

echo "=========================================="
echo "MCP Server Deployment"
echo "=========================================="

# Configuration
AWS_REGION="us-east-1"
MCP_STACK_NAME="mcp-package-lifecycle-predictor"
SAGEMAKER_ENDPOINT_NAME="mcp-package-lifecycle-predictor"
NEPTUNE_ENDPOINT="swa-shipgraph-neptune-instance-prod-us-east-1-read-replica.c6fskces27nt.us-east-1.neptune.amazonaws.com:8182/gremlin"
VPC_ID="vpc-051f3960cd930799d"
PRIVATE_SUBNET_IDS="subnet-0cca267b3bc77b041,subnet-059c480a9ec1404e1,subnet-0a6d786023e20ba3c"
NEPTUNE_SECURITY_GROUP_ID="sg-084496408d4baa6fa"

echo ""
echo "Step 1: Verify source files..."
cd mcp_server

echo "Files in src/:"
ls -la src/

# Verify all required files exist
REQUIRED_FILES=("lambda_handler.py" "prediction_service.py" "neptune_client.py" "skeleton_builder.py" "requirements.txt")
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "src/$file" ]; then
        echo "ERROR: Missing required file: src/$file"
        exit 1
    fi
    echo "  ✓ $file"
done

echo ""
echo "Step 2: Clean previous build..."
rm -rf .aws-sam

echo ""
echo "Step 3: SAM build..."
sam build

echo ""
echo "Step 4: Verify build output..."
echo "Files in build:"
ls -la .aws-sam/build/MCPServerFunction/

# Check if all Python files are in build
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f ".aws-sam/build/MCPServerFunction/$file" ]; then
        echo "WARNING: $file not in build, copying..."
        cp "src/$file" ".aws-sam/build/MCPServerFunction/"
    fi
done

echo ""
echo "Final build contents:"
ls -la .aws-sam/build/MCPServerFunction/

echo ""
echo "Step 5: SAM deploy..."
sam deploy \
    --stack-name "${MCP_STACK_NAME}" \
    --region "${AWS_REGION}" \
    --resolve-s3 \
    --capabilities CAPABILITY_IAM \
    --parameter-overrides \
        "SageMakerEndpointName=${SAGEMAKER_ENDPOINT_NAME}" \
        "NeptuneEndpoint=${NEPTUNE_ENDPOINT}" \
        "VpcId=${VPC_ID}" \
        "SubnetIds=${PRIVATE_SUBNET_IDS}" \
        "NeptuneSecurityGroupId=${NEPTUNE_SECURITY_GROUP_ID}" \
    --no-confirm-changeset \
    --no-fail-on-empty-changeset

cd ..

echo ""
echo "Step 6: Verify deployment..."

# Download and check deployed code
echo "Checking deployed Lambda code..."
LAMBDA_URL=$(aws lambda get-function \
    --function-name mcp-package-lifecycle-predictor-server \
    --region us-east-1 \
    --query 'Code.Location' \
    --output text)

curl -s -o /tmp/deployed.zip "$LAMBDA_URL"
echo "Deployed package contents:"
unzip -l /tmp/deployed.zip | grep ".py"

# Get outputs
MCP_ENDPOINT=$(aws cloudformation describe-stacks \
    --stack-name "${MCP_STACK_NAME}" \
    --query 'Stacks[0].Outputs[?OutputKey==`MCPApiEndpoint`].OutputValue' \
    --output text \
    --region "${AWS_REGION}")

echo ""
echo "=========================================="
echo "Deployment Complete!"
echo "=========================================="
echo ""
echo "MCP API Endpoint: ${MCP_ENDPOINT}"
echo ""
echo "Test command:"
echo "  curl -X POST '${MCP_ENDPOINT}' -H 'Content-Type: application/json' -d '{\"jsonrpc\":\"2.0\",\"id\":\"1\",\"method\":\"tools/list\",\"params\":{}}'"