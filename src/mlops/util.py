from google.cloud import storage

def save_to_gcs(local_path, gcs_path):
    """Uploads files to Google Cloud Storage."""
    client = storage.Client()
    bucket_name, gcs_blob_path = gcs_path.replace("gs://", "").split("/", 1)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_blob_path)
    blob.upload_from_filename(local_path)
