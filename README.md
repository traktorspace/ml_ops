![py-version](https://databadges.traktor.space/badge/Python-3.12-blue)
![last-commit](https://databadges.traktor.space/github/last-commit/traktorspace/ml_ops)
![contributors](https://databadges.traktor.space/github/contributors/traktorspace/ml_ops)

# ML Ops
- [What is ML Ops?](#what-is-ml-ops)
  - [How it looks in this repository](#how-it-looks-in-this-repository)
- [Installation \& Setup](#installation--setup)
  - [1. Dependencies](#1-dependencies)
    - [Option 1-A - Local installation](#option-1-a---local-installation)
    - [Option 2-A - Dockerized installation](#option-2-a---dockerized-installation)
  - [2. Cloud setup](#2-cloud-setup)
  - [3. Pipeline setup](#3-pipeline-setup)

## What is ML Ops?

MLOps (Machine-Learning Operations) adapts the proven ideas of DevOps-version control, automated testing, CI/CD, monitoring-to the life-cycle of data-driven models.  It treats datasets, training code, model artefacts and evaluation metrics as first-class, versioned entities, so that every change (be it a new batch of data or a tweak in hyper-parameters) can be reproduced, audited and rolled back just like a software release.

A typical MLOps pipeline therefore automates three feedback loops:

1. Data loop – discover, ingest, validate and label fresh data.  
2. Training loop – re-train and hyper-tune the model whenever the data or code changes.  
3. Deployment/monitoring loop – ship the model, watch it in production, and trigger the next cycle when performance drifts.

### How it looks in this repository  

In our case the main bottleneck is not code but data availability and annotation throughput.  The pipeline is designed to:

1. Periodically fetch raw product images from the “data-archive” GCS bucket, normalise the folder structure and push the metadata to Encord so that annotators can start working immediately.  
2. Poll Encord for newly finished annotations; once **N** fresh samples (the “X amount”) are ready, pull them back into the repository, run data-quality checks and add them to the training set.  
3. Launch a new training run and an automatic evaluation against the previous best model.  If the key metrics improve (or regress within an acceptable margin while adding valuable classes), the CI/CD workflow promotes the model to the “staging” or “prod” registry tag; otherwise it raises a Slack alert for manual review.

By codifying each step—data extraction, format conversion, annotation sync, training, evaluation and model promotion—in version-controlled scripts and workflows, we get a repeatable, auditable and fully automated ML supply chain that scales with both data volume and team size.

## Installation & Setup

This repo uses [uv](https://docs.astral.sh/uv/getting-started/installation/) as package manager, please verify to have it already installed.

### 1. Dependencies

#### Option 1-A - Local installation
1.  Create a virtualenv with `uv` using
    ```
    uv venv
    ```

2. Activate the env
    ```
    source .venv/bin/activate
    ```
3. Install all the dependencies
    ```
    uv sync
    ``` 
4. Install the `pre-commit` hook 
    ```
    pre-commit install 
    ``` 

#### Option 2-A - Dockerized installation
1. Install [podman](https://podman.io/docs/installation)
2. Build the container
    ```shell
    podman build --ssh default=$SSH_AUTH_SOCK -t mlops .
    ```

    **Note:** In case off ssh problems try
    ```shell
    # Start SSH agent (if not already running)
    eval "$(ssh-agent -s)"

    # Add your SSH key
    ssh-add ~/.ssh/id_ed25519

    # Check SSH agent has keys
    ssh-add -l

    # Verify $SSH_AUTH_SOCK is set
    echo $SSH_AUTH_SOCK
    ```
3. WIP


### 2. Cloud setup

The images can be uploaded in Encord in many ways, the one used in this project is through the [Encord GCP Integration](https://docs.encord.com/platform-documentation/General/annotate-data-integrations/annotate-gcp-integration). In this way it's possible to let Encord know where the images are located in your bucket. [This is an example](https://docs.encord.com/platform-documentation/Index/add-files/index-register-cloud-data#image-groups) of how Encord expects these files declared for image groups.

Before proceeding be sure that you have access to all the buckets you need.\
For annotation-related buckets please contact either _Tommaso Canova_ or _Arthur Vandenhoeke_, or in case of products-only related buckets please contact either _Lennert Antson_ or _Matti Bragge_.

1. If you don't have it already, install the [Google Cloud CLI](https://cloud.google.com/sdk/docs/install-sdk)
2. Authenticate using:
    ```
    gcloud auth application-default login
    ```

### 3. Pipeline setup
Before running an annotation pushing/pulling script, it's necessary to have a `.env` file with all the necessary credentials that looks like this:

```shell
# Encord 
ENCORD_PRIVATE_KEY_PATH=...
ENCORD_CLOUD_PROJECT_HASH=...
# Slack
SLACK_OAUTH=...
SLACK_CHANNEL=...
# Postgres 
POSTGRES_DB_HOST=...
POSTGRES_DB_PORT=...
POSTGRES_DB_NAME=...
POSTGRES_DB_USER=...
POSTGRES_DB_PASS=...
# Google cloud - Annotation bucket
ANNOTATIONS_GOOGLE_CLOUD_PROJECT=...
ANNOTATIONS_GOOGLE_CLOUD_BUCKET_NAME=...
# Google cloud - Products data archive
DATA_ARCHIVE_GOOGLE_CLOUD_PROJECT=...
DATA_ARCHIVE_GOOGLE_CLOUD_BUCKET_NAME=...
```

| Environment variable                    | Where to get the value                                                                                    | What it represents / why it’s needed                                              |
| --------------------------------------- | --------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| `ENCORD_PRIVATE_KEY_PATH`               | [Docs](https://docs.encord.com/platform-documentation/Annotate/annotate-sdk-keys)                         | Absolute/relative path used by the SDK to authenticate against Encord’s API.      |
| `ENCORD_CLOUD_PROJECT_HASH`             | Copy the Project ID Hash from: Annotate > Projects > your-project > Project ID                            | Unique identifier of the Encord project the pipeline should write annotations to. |
| `SLACK_OAUTH`                           | [Slack API](https://api.slack.com/apps) → Your App → OAuth & Permissions → “Bot User OAuth Token”.        | Token the app uses to post messages to Slack.                                     |
| `SLACK_CHANNEL`                         | Open the Slack channel → channel details → “Copy channel ID” (or use `/who` then click the channel info). | Channel ID that will receive build / ingestion / alert notifications.             |
| `POSTGRES_DB_HOST`                      | Bitwarden                                                                                                 | Hostname or IP of the Postgres instance the service connects to.                  |
| `POSTGRES_DB_PORT`                      | Bitwarden                                                                                                 | TCP port Postgres is listening on.                                                |
| `POSTGRES_DB_NAME`                      | Bitwarden                                                                                                 | Logical database the service should use.                                          |
| `POSTGRES_DB_USER`                      | Bitwarden                                                                                                 | Username used in the Postgres connection string.                                  |
| `POSTGRES_DB_PASS`                      | Bitwarden                                                                                                 | Password for `POSTGRES_DB_USER`.                                                  |
| `ANNOTATIONS_GOOGLE_CLOUD_PROJECT`      | Google Cloud Console → Home → Project info (project ID).                                                  | GCP project that owns the bucket where generated annotation files are stored.     |
| `ANNOTATIONS_GOOGLE_CLOUD_BUCKET_NAME`  | Storage → Browser → select bucket → bucket name.                                                          | Name of the GCS bucket that will contain exported annotations.                    |
| `DATA_ARCHIVE_GOOGLE_CLOUD_PROJECT`     | Google Cloud Console → Project selector.                                                                  | GCP project holding the source data-archive bucket.                               |
| `DATA_ARCHIVE_GOOGLE_CLOUD_BUCKET_NAME` | Storage → Browser → bucket name.                                                                          | Name of the bucket where the archived product data lives.                         |


