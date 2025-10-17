This is a small glimpse of a project I built during my internship. It’s a little similar to—but not the same as—my original implementation.
# Face Recognition → MongoDB Pipeline

Lightweight helper script that extracts face embeddings from an image using the `face_recognition` library and stores those vectors inside a MongoDB collection.

## 1. Prerequisites
- Python 3.9+ recommended (the `face_recognition` package depends on `dlib`)
- A running MongoDB instance (local or hosted)
- System packages required by `dlib` (on Ubuntu: `sudo apt-get install build-essential cmake libopenblas-dev liblapack-dev libx11-dev libgtk-3-dev`)

## 2. Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional: create a `.env` file so the script can load connection details automatically.
```bash
MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=face_recognition
MONGODB_COLLECTION=embeddings
```

## 3. CLI usage
```bash
python -m src.face_store --image path/to/photo.jpg --label "Person Name"
```

Key flags:
- `--image`: path to the image that contains at least one face (required)
- `--label`: optional label; defaults to `filename#index` if omitted
- `--mongo-uri`, `--database`, `--collection`: override the Mongo target; each falls back to the respective env var or to the default shown above
- `--dry-run`: print the embeddings JSON to stdout instead of writing to MongoDB (useful for testing)

Example (dry run):
```bash
python -m src.face_store --image samples/alice.jpg --label Alice --dry-run
```

Example (store in MongoDB):
```bash
python -m src.face_store --image samples/alice.jpg --label Alice
```

Each stored document looks like:
```json
{
  "label": "Alice",
  "embedding": [0.10302, -0.05641, ...],
  "source_image": "/abs/path/to/samples/alice.jpg",
  "created_at": "2024-04-12T15:34:21.123456+00:00",
  "metadata": {"face_index": 0}
}
```

## 4. Flask API
Start the development server (loads `.env` automatically):
```bash
flask --app src.app run --debug
```

Send an image using `curl`:
```bash
curl -X POST http://127.0.0.1:5000/faces \
  -F "image=@samples/alice.jpg" \
  -F "label=Alice"
```

Response:
```json
{
  "inserted_ids": ["661985c5792f3f2aa2b0480b"],
  "faces_detected": 1,
  "label": "Alice"
}
```

Health check endpoint:
```bash
curl http://127.0.0.1:5000/health
```

## 5. Verifying the data
Use the Mongo shell or Compass to confirm the saved vector:
```bash
mongosh
use face_recognition
db.embeddings.find().pretty()
```

## 6. Troubleshooting
- **No faces detected**: ensure the image has clear, front-facing faces; try `face_recognition.face_locations(..., model="cnn")` if you have GPU support.
- **Import errors for `dlib`**: confirm the system dependencies above are installed before `pip install`.
