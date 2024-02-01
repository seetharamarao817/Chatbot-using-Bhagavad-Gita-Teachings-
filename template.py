import os
from pathlib import Path


list_of_files = [
   "src/__init__.py",
   "src/components/__init__.py", 
   "src/components/data_ingestion.py", 
   "src/exceptions/__init__.py",
   "src/exceptions/exception.py",
   "src/logging/logger.py",
   "src/logging/__init__.py",
   "src/utils/__init__.py",
   "src/utils/utils.py",
   "requirements.txt", 
   "requirements_dev.txt",
   "pipelines/__init__.py",
   "pipelines/training_pipeline.py",
   "pipelines/prediction_pipeline.py",
   "setup.py",
   "init_setup.sh"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass # create an empty file