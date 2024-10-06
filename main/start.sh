jupyter nbconvert --to script trainer.ipynb

PROJECT_ID="mlops-433612"
REPO_NAME='custom-training'
IMAGE_URI=us-central1-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/training_image:latest

docker build ./ -t $IMAGE_URI

docker push $IMAGE_URI

python3 job.py
