"""
Flask application exposing endpoints to generate face embeddings and store them in MongoDB.
"""

from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from werkzeug.exceptions import BadRequest

from .face_store import get_collection, load_image_encodings, persist_embeddings


def _resolve_collection():
    """
    Lazily create the MongoDB collection handle using environment variables.
    """
    mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    database = os.getenv("MONGODB_DATABASE", "face_recognition")
    collection = os.getenv("MONGODB_COLLECTION", "embeddings")
    return get_collection(mongo_uri, database, collection)


def create_app() -> Flask:
    load_dotenv()
    app = Flask(__name__)
    collection = _resolve_collection()

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"}), 200

    @app.route("/faces", methods=["POST"])
    def store_face() -> tuple[dict[str, object], int]:
        uploaded = request.files.get("image")
        if uploaded is None:
            raise BadRequest("Missing 'image' file in form-data.")

        label: Optional[str] = request.form.get("label")
        source_name = uploaded.filename or label or "uploaded-image"

        try:
            embeddings = load_image_encodings(uploaded)
        except FileNotFoundError as exc:
            raise BadRequest(str(exc)) from exc
        except ValueError as exc:
            raise BadRequest(str(exc)) from exc

        inserted_ids = persist_embeddings(collection, embeddings, label, source_name)
        return (
            jsonify(
                {
                    "inserted_ids": inserted_ids,
                    "faces_detected": len(inserted_ids),
                    "label": label,
                }
            ),
            201,
        )

    return app


app = create_app()
