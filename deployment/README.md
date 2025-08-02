# Hate Speech Detection API - Deployment

## Local Testing

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the API:
```bash
python app.py
```

3. Test the API:
```bash
python test_api.py
```

## Render Deployment

1. Push code to GitHub
2. Go to render.com and create new Web Service
3. Connect your GitHub repository
4. Select the deployment folder
5. Configure:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120`
   - Environment: Python 3
6. Deploy

## Vercel Deployment

1. Install Vercel CLI:
```bash
npm i -g vercel
```

2. Deploy:
```bash
vercel
```

3. Follow prompts and select deployment folder

## API Endpoints

- `GET /` - API documentation
- `GET /health` - Health check
- `POST /predict` - Single text prediction
- `POST /predict-batch` - Multiple texts prediction

## Example Usage

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text here"}'
``` 