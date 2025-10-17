"""
Utility script for extracting facial embeddings from an image and saving them to MongoDB.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import BinaryIO, List, Optional, Union

import face_recognition
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.collection import Collection


ImageSource = Union[str, BinaryIO]


def _normalize_stream(image_source: ImageSource) -> ImageSource:
    """
    Prepare the image source for loading. If we receive a stream, rewind it so face_recognition can read it.
    """
    if isinstance(image_source, str):
        return image_source

    stream = getattr(image_source, "stream", image_source)
    if hasattr(stream, "seek"):
        stream.seek(0)
    return stream


def load_image_encodings(image_source: ImageSource) -> List[List[float]]:
    """
    Load an image (path or binary stream) and return facial embeddings for each face detected.
    """
    if isinstance(image_source, str) and not os.path.isfile(image_source):
        raise FileNotFoundError(f"Image not found: {image_source}")

    normalized_source = _normalize_stream(image_source)
    image = face_recognition.load_image_file(normalized_source)
    # Model can be 'hog' (CPU) or 'cnn' (GPU); hog is smaller/slower but portable.
    face_locations = face_recognition.face_locations(image, model="hog")
    if not face_locations:
        raise ValueError("No faces detected in the provided image.")

    return face_recognition.face_encodings(image, face_locations)


def get_collection(mongo_uri: str, database: str, collection: str) -> Collection:
    """
    Instantiate a MongoDB collection handle.
    """
    client = MongoClient(mongo_uri)
    return client[database][collection]


def persist_embeddings(
    collection: Collection,
    embeddings: List[List[float]],
    label: Optional[str],
    source_image: str,
) -> List[str]:
    """
    Save embeddings to MongoDB and return the inserted IDs as strings.
    """
    now = datetime.now(timezone.utc)
    documents = []
    for idx, encoding in enumerate(embeddings):
        documents.append(
            {
                "label": label or f"{os.path.basename(source_image)}#{idx}",
                "embedding": encoding,
                "source_image": os.path.abspath(source_image),
                "created_at": now,
                "metadata": {"face_index": idx},
            }
        )

    result = collection.insert_many(documents)
    return [str(inserted_id) for inserted_id in result.inserted_ids]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract facial embeddings from an image and store them in MongoDB."
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the image file that contains at least one face.",
    )
    parser.add_argument(
        "--label",
        help="Optional label to associate with the embedding(s). Defaults to filename#index.",
    )
    parser.add_argument(
        "--mongo-uri",
        default=os.getenv("MONGODB_URI", "mongodb://localhost:27017"),
        help="MongoDB connection URI, defaults to MONGODB_URI env var or localhost.",
    )
    parser.add_argument(
        "--database",
        default=os.getenv("MONGODB_DATABASE", "face_recognition"),
        help="Target MongoDB database name (default: face_recognition).",
    )
    parser.add_argument(
        "--collection",
        default=os.getenv("MONGODB_COLLECTION", "embeddings"),
        help="Target MongoDB collection name (default: embeddings).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="If set, print the embeddings to stdout instead of writing to MongoDB.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    embeddings = load_image_encodings(args.image)

    if args.dry_run:
        payload = [
            {
                "label": args.label or f"{os.path.basename(args.image)}#{idx}",
                "embedding": encoding,
                "source_image": os.path.abspath(args.image),
            }
            for idx, encoding in enumerate(embeddings)
        ]
        print(json.dumps(payload, indent=2))
        return

    collection = get_collection(args.mongo_uri, args.database, args.collection)
    inserted_ids = persist_embeddings(collection, embeddings, args.label, args.image)
    print("Inserted documents:", inserted_ids)


if __name__ == "__main__":
    main()
