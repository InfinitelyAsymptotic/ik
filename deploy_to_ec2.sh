#!/bin/bash
# Part 4: Deployment Scripts for EC2
# File: deploy_to_ec2.sh

# This script contains commands to deploy your ML model to EC2
# Run these commands step by step in your terminal

echo "🚀 ML Model Deployment to EC2 Guide"
echo "Run these commands step by step:"

echo ""
echo "1️⃣ PREPARE LOCAL FILES FOR DEPLOYMENT"
echo "# Create deployment package"
echo "mkdir ml_deployment"
echo "cp app.py ml_deployment/"
echo "cp requirements.txt ml_deployment/"
echo "cp -r saved_models ml_deployment/"

echo ""
echo "2️⃣ COPY FILES TO EC2"
echo "# Replace your-key.pem and your-ec2-ip with actual values"
echo "scp -i your-key.pem -r ml_deployment ubuntu@your-ec2-ip:~/"

echo ""
echo "3️⃣ SSH INTO EC2 AND SETUP"
echo "ssh -i your-key.pem ubuntu@your-ec2-ip"

echo ""
echo "4️⃣ COMMANDS TO RUN ON EC2 (after SSH):"

cat << 'EOF'
# Update system
sudo apt update
sudo apt install -y python3 python3-pip python3-venv

# Create virtual environment
cd ml_deployment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Test the app locally on EC2
python app.py

# Run with uvicorn for production
uvicorn app:app --host 0.0.0.0 --port 8000

# Or run in background
nohup uvicorn app:app --host 0.0.0.0 --port 8000 > app.log 2>&1 &
EOF

echo ""
echo "5️⃣ CONFIGURE EC2 SECURITY GROUP"
echo "# Add inbound rule for port 8000:"
echo "# - Type: Custom TCP"
echo "# - Port: 8000"
echo "# - Source: 0.0.0.0/0 (or your IP for security)"

echo ""
echo "6️⃣ TEST DEPLOYMENT"
echo "# Replace your-ec2-ip with actual IP"
echo "curl http://your-ec2-ip:8000/ping"

echo ""
echo "✅ Your model should now be accessible at: http://your-ec2-ip:8000"
