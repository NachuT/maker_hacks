#!/bin/bash

echo "🚀 Uploading Stress Less to GitHub..."

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "❌ Please run this script from the project root directory"
    exit 1
fi

# Create a temporary directory for the upload
TEMP_DIR="/tmp/stress_less_upload"
rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR"
cd "$TEMP_DIR"

echo "📁 Cloning repository..."
git clone https://github.com/NachuT/maker_hacks.git .
if [ $? -ne 0 ]; then
    echo "❌ Failed to clone repository. Please check the URL and permissions."
    exit 1
fi

echo "📋 Copying files..."
# Copy all the important files
cp /Users/aditya/hhgb/app.py .
cp /Users/aditya/hhgb/scaler.pkl .
cp /Users/aditya/hhgb/hgb_model.pkl .
cp /Users/aditya/hhgb/package.json .
cp /Users/aditya/hhgb/next.config.js .
cp /Users/aditya/hhgb/tsconfig.json .
cp /Users/aditya/hhgb/postcss.config.js .
cp -r /Users/aditya/hhgb/src .
cp -r /Users/aditya/hhgb/public .

echo "📝 Adding files to git..."
git add .

echo "💾 Committing changes..."
git commit -m "Stress Less - Real-time health monitoring app with Arduino integration

Features:
- Live heart rate monitoring with real-time updates
- Breathing exercises with adaptive patterns
- Health statistics and environmental monitoring
- Stress management tips and emergency relief
- Arduino/ESP32 integration via HTTP POST
- AI-powered stress prediction using machine learning
- Next.js frontend with real-time data streaming
- Local CSV-based authentication system

Setup:
1. npm install
2. pip install flask torch scikit-learn numpy joblib
3. python3 app.py (Terminal 1)
4. npm run dev (Terminal 2)
5. Visit http://localhost:3000

Arduino Integration:
Send data to http://YOUR_MAC_IP:5010/api/data
{\"hb\": 78, \"temp\": 22.5, \"hum\": 45.0}"

echo "🚀 Pushing to GitHub..."
git push origin main

if [ $? -eq 0 ]; then
    echo "✅ Successfully uploaded to GitHub!"
    echo "🌐 Repository: https://github.com/NachuT/maker_hacks"
else
    echo "❌ Failed to push to GitHub. You may need to:"
    echo "1. Set up GitHub authentication (git config --global user.name and user.email)"
    echo "2. Use a personal access token"
    echo "3. Upload manually via GitHub web interface"
fi

# Clean up
cd /Users/aditya/hhgb
rm -rf "$TEMP_DIR"

echo "🏁 Upload process complete!"
