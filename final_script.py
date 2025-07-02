import boto3
import csv

# AWS Rekognition client
rekognition = boto3.client('rekognition', region_name='us-east-1')

# S3 bucket and image keys
bucket = 'aeroxplorer-rekognition'
image_keys = [
    'imagesCUBE/airplane1.jpg',
    'imagesCUBE/airplane2.jpeg'
    # Add more image paths as needed
]

# Prepare output rows
rows = [("Filename", "Detected Labels", "Detected Text")]

# Process each image
for image_key in image_keys:
    print(f"--- Processing: {image_key} ---")

    label_resp = rekognition.detect_labels(
        Image={'S3Object': {'Bucket': bucket, 'Name': image_key}},
        MaxLabels=15,
        MinConfidence=70
    )
    labels = [label['Name'] for label in label_resp['Labels']]
    print("Labels:", labels)

    text_resp = rekognition.detect_text(
        Image={'S3Object': {'Bucket': bucket, 'Name': image_key}}
    )
    lines = [
        text['DetectedText'] 
        for text in text_resp['TextDetections'] 
        if text['Type'] == 'LINE'
    ]
    print("Text Detected:", lines)

    rows.append((image_key, ", ".join(labels), ", ".join(lines)))

# Write results to CSV
with open("rekognition_tags_output.csv", "w", newline="") as f:
    csv.writer(f).writerows(rows)

print("rekognition_tags_output.csv saved.")
