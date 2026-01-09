#!/bin/bash
set -e

echo "Packaging Model for SageMaker (inference only)..."

MODEL_DIR="model_package"
OUTPUT_FILE="model.tar.gz"

rm -rf ${MODEL_DIR} ${OUTPUT_FILE}
mkdir -p ${MODEL_DIR}/code

# Copy model artifacts
cp ../trained_model/best_model.pt ${MODEL_DIR}/
cp ../trained_model/preprocessor.pkl ${MODEL_DIR}/

# Copy inference code
cp code/inference.py ${MODEL_DIR}/code/
cp code/requirements.txt ${MODEL_DIR}/code/
cp code/config.py ${MODEL_DIR}/code/
cp -r code/models ${MODEL_DIR}/code/
cp -r code/data ${MODEL_DIR}/code/

# Create tarball
cd ${MODEL_DIR}
tar -czvf ../${OUTPUT_FILE} .
cd ..

echo "Created: ${OUTPUT_FILE} ($(du -h ${OUTPUT_FILE} | cut -f1))"