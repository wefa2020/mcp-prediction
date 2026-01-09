# 1. Update configuration in scripts/deploy_all.sh

# 2. Copy your model files
cp -r /path/to/checkpoints sagemaker/
cp /path/to/config.py sagemaker/code/
cp -r /path/to/models sagemaker/code/
cp -r /path/to/data sagemaker/code/

# 3. Deploy everything
chmod +x scripts/deploy_all.sh
./scripts/deploy_all.sh

# 4. Test
python scripts/test_mcp.py --endpoint https://xxx.execute-api.us-east-1.amazonaws.com/prod/mcp