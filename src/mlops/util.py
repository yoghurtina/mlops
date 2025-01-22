from google.cloud import storage
import os

def save_to_gcs(local_path, gcs_path):
    """Uploads files to Google Cloud Storage."""
    client = storage.Client()
    bucket_name, gcs_blob_path = gcs_path.replace("gs://", "").split("/", 1)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_blob_path)
    blob.upload_from_filename(local_path)

def download_from_gcs(bucket_name: str, gcs_path: str, local_dir: str):
    """
    Download a model directory from Google Cloud Storage to a local directory.

    Args:
        bucket_name (str): Name of the GCS bucket.
        gcs_path (str): Path to the model directory in the bucket.
        local_dir (str): Local directory to save the model files.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=gcs_path)

    os.makedirs(local_dir, exist_ok=True)

    for blob in blobs:
        # Remove the prefix from the GCS path to get the local relative path
        local_file_path = os.path.join(local_dir, blob.name[len(gcs_path):].lstrip("/"))
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        blob.download_to_filename(local_file_path)
