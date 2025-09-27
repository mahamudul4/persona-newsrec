#!/bin/bash

while getopts ":e:r:" flag; do
    case "${flag}" in
    e) env=${OPTARG} ;;
    r) region=${OPTARG} ;;
    *)
        echo "Invalid option. Only -e and -r are allowed" >&2
        exit 1
        ;;
    esac
done
env=${env:-prod}
region=${region:-us-east-1}

echo "ENV: $env"
echo "Region: $region"

# Skip DVC pull if models already exist locally
if [ ! -d "models/nrms-mind" ] || [ ! -d "models/distilbert-base-uncased" ]; then
    echo "Models not found locally, attempting to pull from DVC..."
    dvc pull -R models || exit 2
else
    echo "Models found locally, skipping DVC pull"
fi

#dvc pull -R models || exit 2
# Build container and deploy functions
npx serverless deploy --stage "${env}" --region "${region}" || exit 3
