import dotenv
from google.cloud import bigquery

from gcpUtils import create_big_query_dataset, create_big_query_table

dotenv.load_dotenv()

project_id = dotenv.get_key(".env", "GOOGLE_CLOUD_PROJECT")
dataset_id = "fineweb_edu_classification"
table_id = "classifier_outputs"


# create_big_query_dataset(project_id, dataset_id, location="europe-west1")


create_big_query_table(project_id, dataset_id, table_id, schema)
