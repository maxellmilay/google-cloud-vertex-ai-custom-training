from google.cloud import aiplatform
import os

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "key.json"

PROJECT_ID="mlops-433612"

REPO_NAME='custom-training'

LOCATION = 'us-central1'

aiplatform.init(
    project=PROJECT_ID,
    location=LOCATION
)

BUCKET = f'{PROJECT_ID}-{REPO_NAME}/staging'

print('\nCreating training job...\n')

my_job = aiplatform.CustomContainerTrainingJob(display_name='custom-job',
                                               project=PROJECT_ID,
                                               container_uri=f'us-central1-docker.pkg.dev/{PROJECT_ID}/{REPO_NAME}/training_image:latest',
                                               staging_bucket=f'gs://{BUCKET}')


print('Running training job...\n')

my_job.run(replica_count=1,
           machine_type='n1-highcpu-16')

print('TRAINING JOB SUCCESSFUL')
