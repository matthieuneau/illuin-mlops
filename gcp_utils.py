from google.cloud import storage


def upload_to_bucket(bucket_name, source_file_path, destination_blob_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_path)


upload_to_bucket("fineweb-datasets", "test.txt", "test.txt")
