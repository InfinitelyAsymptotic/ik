# AWS Free Tier Registration Guide for MLOps Workshop

## Overview
This guide will help you register for AWS Free Tier to deploy FastAPI ML models on EC2 instances. The Free Tier provides 12 months of free access to key AWS services with usage limits that are perfect for learning MLOps deployment.

## What You'll Get with AWS Free Tier

### For Your MLOps Workshop:
- **750 hours/month** of EC2 t2.micro instances (enough to run 1 instance 24/7)
- **30 GB** of Amazon EBS storage
- **5 GB** of Amazon S3 storage
- **750 hours/month** of Elastic Load Balancing
- **1 million** API Gateway requests/month
- **CloudWatch** monitoring (10 metrics, 10 alarms)

## Prerequisites
- Valid email address
- Credit/debit card (for identity verification - won't be charged for Free Tier usage)
- Phone number for verification
- Valid government-issued ID may be required

## Step-by-Step Registration Process

### Step 1: Start Registration
1. Go to [aws.amazon.com](https://aws.amazon.com)
2. Click **"Create an AWS Account"** (orange button in top right)
3. You'll be redirected to the sign-up page

### Step 2: Create Root User Account
1. **Email Address**: Enter your email address
2. **AWS Account Name**: Choose a descriptive name (e.g., "YourName-MLOps-Learning")
3. Click **"Verify email address"**
4. Check your email and enter the verification code
5. Create a strong password (minimum 8 characters, mix of letters, numbers, symbols)
6. Confirm password
7. Click **"Continue"**

### Step 3: Contact Information
1. **Account Type**: Select **"Personal"** (unless you're registering for a company)
2. Fill in your personal information:
   - Full name
   - Phone number
   - Country/Region
   - Address
   - City
   - State/Province
   - Postal code
3. **AWS Customer Agreement**: Read and check the box to agree
4. Click **"Create Account and Continue"**

### Step 4: Payment Information
1. Enter your credit/debit card information
   - Card number
   - Expiration date
   - Cardholder name
   - Security code
2. **Billing Address**: Choose "Use the address I entered above" or enter different billing address
3. Click **"Verify and Continue"**

> **Important**: Your card will only be charged if you exceed Free Tier limits. For this workshop, staying within limits means no charges.

### Step 5: Identity Verification
1. **Phone Verification**:
   - Enter your phone number
   - Choose verification method: "Text message (SMS)" or "Voice call"
   - Click **"Send SMS"** or **"Call me now"**
2. Enter the 4-digit verification code you receive
3. Click **"Verify Code and Continue"**

### Step 6: Choose Support Plan
1. Select **"Basic Support - Free"** (recommended for learning)
   - Includes 24/7 access to customer service
   - Documentation and whitepapers
   - Community forums
2. Click **"Complete Sign Up"**

### Step 7: Confirmation
1. You'll see a confirmation page: "Congratulations! Your AWS account is ready"
2. Click **"Go to the AWS Management Console"**
3. Sign in with your root user credentials

## Post-Registration Setup

### Step 1: Secure Your Account
1. **Enable MFA (Multi-Factor Authentication)**:
   - Go to IAM Console → Root user → Security credentials
   - Enable MFA using your phone's authenticator app
   
2. **Create IAM User** (Don't use root user for daily tasks):
   - Go to IAM Console → Users → Create user
   - Username: `mlops-learner` (or similar)
   - Attach policy: `PowerUserAccess` or `AdministratorAccess`
   - Create access keys for programmatic access

### Step 2: Verify Free Tier Eligibility
1. Go to [AWS Account Activity](http://aws.amazon.com/account/)
2. Click **"Account Activity"**
3. Look for the Free Tier notification at the top
4. Should show: "Your AWS Free Tier expires on [date 12 months from now]"

### Step 3: Set Up Billing Alerts
1. Go to **Billing and Cost Management Console**
2. Click **"Billing preferences"**
3. Enable **"Receive Billing Alerts"**
4. Set up CloudWatch billing alarm for $1-5 to monitor usage

## For Your MLOps Workshop Deployment

### EC2 Instance Configuration
When launching your EC2 instance for FastAPI deployment:

1. **AMI**: Choose "Amazon Linux 2" or "Ubuntu Server" (Free Tier eligible - marked with star)
2. **Instance Type**: Select **t2.micro** (only Free Tier eligible type)
3. **Storage**: Stay within 30 GB EBS limit
4. **Security Group**: Configure for FastAPI (port 8000) and SSH (port 22)

### Usage Monitoring Commands
```bash
# Check current month's usage
aws ce get-dimension-values --dimension Key=SERVICE --time-period Start=2024-01-01,End=2024-01-31

# List running instances
aws ec2 describe-instances --query 'Reservations[].Instances[?State.Name==`running`].[InstanceId,InstanceType]'
```

## Free Tier Limits to Remember

| Service | Free Tier Limit | Workshop Usage |
|---------|----------------|----------------|
| EC2 t2.micro | 750 hours/month | 1 instance 24/7 |
| EBS Storage | 30 GB | OS + ML model + data |
| S3 Storage | 5 GB | Model artifacts, data |
| Data Transfer Out | 15 GB/month | API responses |

## Cost Optimization Tips

### During Workshop:
1. **Stop (don't terminate) instances** when not in use
2. **Use EBS-backed instances** to preserve your work
3. **Monitor data transfer** - downloading large datasets counts against limit
4. **Clean up resources** after workshop completion

### Resource Management:
```bash
# Stop instance (preserves data)
aws ec2 stop-instances --instance-ids i-1234567890abcdef0

# Start instance
aws ec2 start-instances --instance-ids i-1234567890abcdef0

# Terminate instance (deletes everything)
aws ec2 terminate-instances --instance-ids i-1234567890abcdef0
```

## Troubleshooting Common Issues

### Account Activation Delays
- Can take up to 24 hours for full activation
- Check spam folder for verification emails
- Contact AWS support if activation exceeds 24 hours

### Payment Method Issues
- Ensure card supports international transactions
- Some prepaid cards may not work
- Try different card if first one fails

### Free Tier Not Showing
- Check account age (must be < 12 months)
- Verify you're in correct AWS region
- Contact AWS support for verification

## After Workshop Cleanup

### Essential Cleanup Steps:
1. **Terminate EC2 instances**
2. **Delete EBS volumes** (if not needed)
3. **Empty S3 buckets** and delete them
4. **Delete CloudWatch logs** if any
5. **Remove IAM users/roles** created for workshop

### Cleanup Commands:
```bash
# List all running instances
aws ec2 describe-instances --query 'Reservations[].Instances[?State.Name==`running`].InstanceId'

# Terminate all instances (be careful!)
aws ec2 terminate-instances --instance-ids $(aws ec2 describe-instances --query 'Reservations[].Instances[?State.Name==`running`].InstanceId' --output text)

# List EBS volumes
aws ec2 describe-volumes --query 'Volumes[?State==`available`].VolumeId'
```

## Support and Resources

### During Workshop:
- **AWS Free Tier FAQ**: [aws.amazon.com/free/free-tier-faqs/](https://aws.amazon.com/free/free-tier-faqs/)
- **AWS Support**: Use "Basic Support" for account/billing issues
- **AWS Documentation**: Comprehensive guides for all services

### Workshop-Specific Resources:
- Keep this guide handy during deployment
- Monitor your Free Tier usage regularly
- Ask instructor before making configuration changes

## Next Steps After Registration

1. **Complete account security setup** (MFA, IAM user)
2. **Familiarize yourself with AWS Console**
3. **Practice launching/stopping t2.micro instances**
4. **Install AWS CLI** on your local machine for the workshop
5. **Join the workshop with your AWS account ready**

---

**Important Reminder**: The Free Tier is available for 12 months from account creation. After the workshop, remember to clean up resources to avoid any charges. Always monitor your usage through the AWS Billing Dashboard.