#!/usr/bin/env python
# coding: utf-8

# # Titanic Survival

# ## Import Libraries

# In[ ]:


import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split


# ## Load Data

# In[ ]:


import os

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "key.json"


# In[ ]:


PROJECT_ID = "mlops-433612"  # @param {type:"string"}
LOCATION = "us-central1"  # @param {type:"string"}

import vertexai

vertexai.init(project=PROJECT_ID, location=LOCATION)


# In[ ]:


from vertexai.resources.preview.feature_store import FeatureGroup

FEATURE_GROUP_ID = "test_feature_group"
DATASET_ID = "featurestore_demo"
TABLE_ID = "cleaned_table"

BQ_TABLE_URI = f'{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}'

feature_groups = FeatureGroup.list()

# Find the feature group with the desired ID
fg = next((fg for fg in feature_groups if fg.name == FEATURE_GROUP_ID), None)

if fg:
    print(f"Feature group found: {fg}")
else:
    print(f"Feature group with ID {FEATURE_GROUP_ID} not found.")


# In[ ]:


from vertexai.resources.preview.feature_store import Feature

feature_names = ["Name","Age","Sex","Fare","Pclass","SibSp","PassengerId","Survived","Parch","Ticket","Cabin","Embarked"]
features = []

for feature in feature_names:
   f: Feature = fg.get_feature(feature)
   features.append(f)


# In[ ]:


import bigframes
import pandas as pd
from vertexai.resources.preview.feature_store import offline_store

# Filter historical data through the timestamp
entity_df = pd.DataFrame(
  data={
    "entity_id": [f"id-{i}" for i in range(0, 182)],
    "timestamp": [pd.Timestamp("2024-09-15 13:24:22.862434 UTC") for i in range(0, 182)],
  },
)

train_df = offline_store.fetch_historical_feature_values(
  entity_df=entity_df,
  features=features,
)


# In[ ]:


# Convert the BigTable dataframe into a Pandas dataframe
train_df = train_df.to_pandas()


# ## Data Wrangling

# ### Dropping Columns

# In[ ]:


train_df = train_df.drop('Cabin', axis=1)


# In[ ]:


train_df = train_df.dropna(subset=['Age', 'Embarked'])


# ### Inspecting Features

# In[ ]:


categorical_features = pd.get_dummies(train_df[['Pclass','Sex','SibSp','Parch']])
numerical_features = train_df[['Age','Fare']]
target_df = train_df[['Survived']]


# In[ ]:


target_column = target_df['Survived'] 


# ### Feature Engineering

# In[ ]:


X = pd.concat([categorical_features], axis=1)
y = target_column


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.2,random_state=123)


# ## Model

# ### Building Model

# In[ ]:


model = XGBClassifier()

# Train the model
model.fit(X_train, y_train)


# ### Training Model

# In[ ]:


from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")


# ## Saving Model (Google Cloud Storage Bucket)

# In[ ]:


import pickle
from google.cloud import storage
import io

REPOSITORY_NAME = 'custom-training'
MODEL_NAME = 'titanic_survival_classifier'
DESTINATION_BLOB_NAME = f'models/{MODEL_NAME}.pkl'

storage_client = storage.Client(project=PROJECT_ID)

BUCKET_NAME = f'{PROJECT_ID}-{REPOSITORY_NAME}'

bucket = storage_client.get_bucket(BUCKET_NAME)

model_buffer = io.BytesIO()
pickle.dump(model, model_buffer)
model_buffer.seek(0)

# Upload the buffer content to a Google Cloud Storage Bucket
blob = bucket.blob(DESTINATION_BLOB_NAME)
blob.upload_from_file(model_buffer, content_type='application/octet-stream')

print(f"Model uploaded to {BUCKET_NAME}/{DESTINATION_BLOB_NAME}")


# ## Register Model in Model Registry

# In[ ]:


from google.cloud import aiplatform

aiplatform.init(project=PROJECT_ID, location=LOCATION)

model_gcs_uri = f'gs://{BUCKET_NAME}/{DESTINATION_BLOB_NAME}'

model = aiplatform.Model.upload(
    display_name=MODEL_NAME,
    artifact_uri=model_gcs_uri,
    serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-24:latest",  # Update this if using a specific serving container
)

