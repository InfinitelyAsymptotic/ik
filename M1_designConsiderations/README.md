# ** M1 Design Considerations AWS-Based Fraud Detection Demo**

## **🎯 Objective**

Build a lightweight end-to-end skeleton of a fraud detection ML system using AWS services to highlight design bottlenecks that MLOps solves.

---

## **✅ Prerequisites**

* AWS account with billing enabled

* IAM user with admin or required permissions

* AWS CLI installed and configured

* Python 3.8+ environment (EC2 or local)

* Kaggle account and API token (kaggle.json)

---

## **🔐 Step 0: Access AWS Console via Nuvepro**

**Goal**: Begin from the Nuvepro-provided cloud lab environment

1. Log in to your Nuvepro account.

2. Navigate to the lab dashboard.

3. Click on the assigned AWS environment.

4. Use provided credentials or direct link to access the AWS Console.



Nuvepro offers sandboxed environments for training and demos. Ensure your session remains active.

---

## **📊 Step 1: Setup S3 Bucket**

**Goal**: Store raw data and simulate ingestion

1. Go to AWS Console \> S3

2. Create bucket: fraud-demo-data-yourname

3. Enable **versioning**

4. **Block all public access**

This bucket is your data lake. Versioning ensures traceability and reproducibility in ML pipelines.

---

## **🚀 Step 2: Launch EC2 Instance (Compute Node)**

**Goal**: Act as the data ingestion \+ training server

1. Launch an EC2 instance with **Ubuntu 22.04**

2. Attach IAM Role with these permissions:

   * AmazonS3FullAccess

   * AmazonSageMakerFullAccess

   * CloudWatchFullAccess

   * AmazonSSMManagedInstanceCore

3. Add tag: Purpose=fraud-demo

**Access options**: \- **SSH**: Create/select key pair, allow port 22 \- **Session Manager**: Ensure IAM role has SSM access and instance is in public subnet





IAM roles provide secure access to services without hardcoding credentials.

---

## **🔄 Step 3: Configure EC2 Environment, Download and Upload Dataset to S3**

→sudo apt update **&&** sudo apt install \-y python3-pip unzip  
→pip3 install kaggle boto3 pandas watchtower  
→mkdir \~/.kaggle

If kaggle not found:

→pip3 install kaggle  
→export PATH\="$PATH:\~/.local/bin"  
→echo 'export PATH="$PATH:\~/.local/bin"' \>\> \~/.bashrc  
→source \~/.bashrc

Upload kaggle.json: \- **SSH**: scp \-i your-key.pem kaggle.json ubuntu@\<EC2-IP\>:\~/.kaggle

**(option) Session Manager**: Use “Upload File” in AWS Console

→chmod 600 \~/.kaggle/kaggle.json



Kaggle CLI needs token authentication and correct permissions to work.

---

## 

**Create Kaggle Account and API Token**

1\. Go to \[https://www.kaggle.com\](https://www.kaggle.com) and sign up or log in.

2\. Click on your profile picture (top-right) \> "Account".

3\. Scroll down to the \*\*API\*\* section.

4\. Click \*\*"Create New API Token"\*\*. This downloads a \`kaggle.json\` file.

5\. Upload this to your EC2 instance under \`\~/.kaggle/\` as described above.

1. Join competition: [IEEE Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection)

2. On EC2:

linux commands:

→kaggle competitions download \-c ieee-fraud-detection  
→unzip ieee-fraud-detection.zip \-d ieee\_fraud  
→aws s3 cp ieee\_fraud/ s3://fraud-demo-data/raw/ \--recursive  
→aws s3 ls s3://fraud-demo-data/raw/(

(in local windows/linux/mac terminal)



Simulates a typical pipeline where ingestion uploads raw data to the cloud.

---

## **📅 Step 5: Basic Data Ingestion \+ Logging Demo**

Run this command to specify region for cloudwatch:

**→export AWS\_DEFAULT\_REGION=us-east-1**

Run this script using editor by running command:

**→vi script.py**

[**Guide to vi editor**](https://www.atmos.albany.edu/daes/atmclasses/atm350/vi_cheat_sheet.pdf)

\#script.py

import boto3, pandas as pd, logging, watchtower

\# Fetch from S3

s3 \= boto3.client('s3')

obj \= s3.get\_object(Bucket='fraud-demo-data-\<yourname\>', Key='raw/train\_transaction.csv')

df \= pd.read\_csv(obj\['Body'\], nrows=5)

print(df)

\# CloudWatch logging with region\_name

logger \= logging.getLogger(\_\_name\_\_)

logger.setLevel(logging.INFO)

logger.addHandler(watchtower.CloudWatchLogHandler(log\_group='fraud-demo-logs'))

logger.info("Loaded 5 rows from S3 successfully")



  


Output on the aws cloudwatch dashboard:  




Logging enables observability. Watchtower helps push logs directly to CloudWatch.

---

## **🛑 Common Errors**

* AccessDenied: IAM role lacks permissions or typo in bucket/key

* NoSuchKey: Wrong file name or missing upload

---

## **🎯 Bottlenecks and Engineering Solutions**

* No dataset versioning → DVC, LakeFS

* No schema validation → Great Expectations, Pandera

* No retraining workflow → GitHub Actions, Jenkins

* Model not tracked → MLflow, SageMaker Model Registry

* No deployment pipeline → EKS, Lambda, SageMaker Endpoint

* No drift detection → EvidentlyAI, WhyLabs

These bottlenecks motivate the need for MLOps practices and tooling.

---

## **🧩 System Design Sketch**

**Raw Data (S3)**  
    **↓**  
**Validation \+ Feature Engineering (EC2/SageMaker)**  
    **↓**  
**Training (SageMaker)**  
    **↓**  
**Model Registry \+ CI/CD (MLflow/Jenkins)**  
    **↓**  
**Deployment (EKS / Lambda / SageMaker Endpoint)**  
    **↓**  
**Monitoring (CloudWatch \+ Drift Detector)**

This architecture illustrates how production ML systems are modular, observable, and automated.

---
