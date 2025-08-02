# Hate Speech Detection API - Render.com Deployment Guide

This guide will help you deploy your Python Hate Speech Detection module to Render.com.

## Prerequisites

1. A Render.com account (free tier available)
2. Your code pushed to a Git repository (GitHub, GitLab, etc.)

## Files Created for Deployment

- `app.py` - Flask web application with API endpoints
- `requirements.txt` - Updated with Flask and web dependencies
- `render.yaml` - Render.com configuration
- `startup.py` - Optional startup script for model initialization

## Deployment Steps

### 1. Push Your Code to Git

Make sure all your files are committed and pushed to your Git repository:

```bash
git add .
git commit -m "Add Flask API for deployment"
git push origin main
```

### 2. Deploy to Render.com

#### Option A: Using render.yaml (Recommended)

1. Go to [Render.com](https://render.com) and sign in
2. Click "New +" and select "Blueprint"
3. Connect your Git repository
4. Render will automatically detect the `render.yaml` file
5. Click "Apply" to deploy

#### Option B: Manual Deployment

1. Go to [Render.com](https://render.com) and sign in
2. Click "New +" and select "Web Service"
3. Connect your Git repository
4. Configure the service:
   - **Name**: `hate-speech-detector-api`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120`
   - **Plan**: Free

### 3. Environment Variables

The following environment variables will be set automatically:
- `PORT`: Set by Render.com
- `PYTHON_VERSION`: 3.9.18

### 4. Model Training

After deployment, you have two options for model training:

#### Option A: Train via API Endpoint

Once deployed, you can train the model using the `/train` endpoint:

```bash
curl -X POST https://your-app-name.onrender.com/train \
  -H "Content-Type: application/json" \
  -d '{"epochs": 5, "batch_size": 32}'
```

#### Option B: Pre-train Locally

Train the model locally and include the model file in your repository:

```bash
python train.py
```

Then commit and push the `hate_speech_detector_lstm.h5` file.

## API Endpoints

Once deployed, your API will be available at `https://your-app-name.onrender.com` with the following endpoints:

### GET `/`
- Returns API documentation and available endpoints

### GET `/health`
- Health check endpoint
- Returns service status and model loading status

### POST `/predict`
- Predict hate speech for a single text
- Body: `{"text": "your text here"}`

### POST `/predict-batch`
- Predict hate speech for multiple texts
- Body: `{"texts": ["text1", "text2", "text3"]}`

### POST `/train`
- Train the model (if dataset is available)
- Body: `{"epochs": 5, "batch_size": 32}` (optional)

## Example Usage

### Single Prediction
```bash
curl -X POST https://your-app-name.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a test message"}'
```

### Batch Prediction
```bash
curl -X POST https://your-app-name.onrender.com/predict-batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Message 1", "Message 2", "Message 3"]}'
```

## Troubleshooting

### Common Issues

1. **Build Fails**: Check that all dependencies are in `requirements.txt`
2. **Model Not Found**: Train the model using the `/train` endpoint
3. **Memory Issues**: The free tier has limited memory. Consider using a smaller model or paid plan
4. **Timeout Issues**: Increase the timeout in the start command if needed

### Logs

Check the logs in your Render.com dashboard for detailed error information.

## Cost Considerations

- **Free Tier**: 750 hours/month, 512MB RAM, shared CPU
- **Paid Plans**: Start at $7/month for more resources

## Security Notes

- The API is public by default
- Consider adding authentication if needed
- Be careful with sensitive data in logs

## Next Steps

After successful deployment:

1. Test all endpoints
2. Monitor performance and logs
3. Consider adding authentication
4. Set up monitoring and alerts
5. Optimize for production use 