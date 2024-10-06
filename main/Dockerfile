FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

RUN apt-get update && apt-get install -y curl \
    && curl https://sdk.cloud.google.com | bash

COPY key.json .

ENV GOOGLE_APPLICATION_CREDENTIALS="key.json"

COPY . .

RUN apt-get update

RUN apt-get install -y apt-transport-https ca-certificates gnupg curl

RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && apt-get update -y && apt-get install google-cloud-cli -y

RUN gcloud auth activate-service-account --key-file=key.json

RUN gcloud config set project mlops-433612

RUN apt-get update && apt-get --only-upgrade install google-cloud-cli-harbourbridge google-cloud-cli-datastore-emulator google-cloud-cli-enterprise-certificate-proxy google-cloud-cli google-cloud-cli-istioctl google-cloud-cli-app-engine-python-extras google-cloud-cli-cbt google-cloud-cli-app-engine-python google-cloud-cli-app-engine-go google-cloud-cli-firestore-emulator google-cloud-cli-bigtable-emulator google-cloud-cli-docker-credential-gcr google-cloud-cli-local-extract google-cloud-cli-app-engine-java google-cloud-cli-anthos-auth google-cloud-cli-cloud-run-proxy google-cloud-cli-pubsub-emulator google-cloud-cli-gke-gcloud-auth-plugin google-cloud-cli-log-streaming google-cloud-cli-kpt google-cloud-cli-kubectl-oidc google-cloud-cli-minikube google-cloud-cli-spanner-migration-tool google-cloud-cli-package-go-module google-cloud-cli-config-connector google-cloud-cli-nomos kubectl google-cloud-cli-anthoscli google-cloud-cli-skaffold google-cloud-cli-terraform-tools google-cloud-cli-spanner-emulator google-cloud-cli-cloud-build-local google-cloud-cli-app-engine-grpc

RUN apt-get install google-cloud-cli

CMD ["python3", "trainer.py"]