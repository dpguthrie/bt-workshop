import csv
import os

import braintrust
from braintrust import Attachment
from dotenv import load_dotenv

load_dotenv()

project_name = os.environ["PROJECT_NAME"]

# Initialize the dataset
dataset = braintrust.init_dataset(project=project_name, name="generic_dataset_images")

# Read and upload the CSV data
with open("datasets/generic_dataset_images.csv", "r") as f:
    reader = csv.DictReader(f)

    for row in reader:
        dataset.insert(
            input=row["query"],
            expected=row["reference_answer"],
            metadata={"image": Attachment(
                data=f"datasets/{row['image']}",
                filename=row['image'],
                content_type="image/jpeg"
            )},
        )

# Flush to ensure all records are uploaded
dataset.flush()

print("Dataset uploaded successfully!")
