import kagglehub

# Download the Dog Emotion Dataset (Cleaned Version) from Kaggle
path = kagglehub.dataset_download("mohitagarwal17/dog-emotion-datasetcleaned-version")

print("Path to dataset files:", path)
