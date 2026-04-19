#!/usr/bin/env bash
set -euo pipefail

BUCKET="${S3_BUCKET:-bucket}"
MODEL_NAME="churn-$(date +%Y%m%d)"
ENDPOINT_NAME="churn-prod"

tar -czf model.tar.gz models/ src/
aws s3 cp model.tar.gz "s3://${BUCKET}/models/churn/model.tar.gz"

aws sagemaker create-model \
  --model-name "${MODEL_NAME}" \
  --primary-container Image="${SAGEMAKER_IMAGE_URI:-123456789012.dkr.ecr.us-east-1.amazonaws.com/churn:latest}",ModelDataUrl="s3://${BUCKET}/models/churn/model.tar.gz" \
  --execution-role-arn "${SAGEMAKER_ROLE_ARN:-arn:aws:iam::123456789012:role/SageMakerExecutionRole}"

aws sagemaker create-endpoint-config \
  --endpoint-config-name "${MODEL_NAME}-config" \
  --production-variants VariantName=AllTraffic,ModelName="${MODEL_NAME}",InitialInstanceCount=2,InstanceType=ml.m5.xlarge,InitialVariantWeight=1.0

aws sagemaker create-endpoint --endpoint-name "${ENDPOINT_NAME}" --endpoint-config-name "${MODEL_NAME}-config"

aws application-autoscaling register-scalable-target \
  --service-namespace sagemaker \
  --resource-id "endpoint/${ENDPOINT_NAME}/variant/AllTraffic" \
  --scalable-dimension sagemaker:variant:DesiredInstanceCount \
  --min-capacity 2 \
  --max-capacity 10

aws application-autoscaling put-scaling-policy \
  --service-namespace sagemaker \
  --resource-id "endpoint/${ENDPOINT_NAME}/variant/AllTraffic" \
  --scalable-dimension sagemaker:variant:DesiredInstanceCount \
  --policy-name churn-target-tracking \
  --policy-type TargetTrackingScaling \
  --target-tracking-scaling-policy-configuration '{"TargetValue":50.0,"PredefinedMetricSpecification":{"PredefinedMetricType":"SageMakerVariantInvocationsPerInstance"}}'

aws cloudwatch put-metric-alarm \
  --alarm-name "${ENDPOINT_NAME}-latency-high" \
  --namespace AWS/SageMaker \
  --metric-name ModelLatency \
  --dimensions Name=EndpointName,Value="${ENDPOINT_NAME}" \
  --statistic Average \
  --period 60 \
  --threshold 100 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 2
