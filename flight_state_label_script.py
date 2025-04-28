import csv

# test Rekognition label outputs for each test image
rekognition_outputs = {
    'test_image1.jpg': ['Runway', 'Airplane', 'Tarmac'],
    'test_image2.jpg': ['Flying', 'Sky'],
    'test_image3.jpg': ['Building', 'Vehicle'],
    'test_image4.jpg': ['Airport', 'Runway'],
    'test_image5.jpg': ['Sky', 'Cloud'],
    'test_image6.jpg': ['Flying', 'Bird'],
    'test_image7.jpg': ['Runway'],
    'test_image8.jpg': ['Airplane', 'Sky', 'Cloud'],
    'test_image9.jpg': ['Tarmac', 'Runway'],
    'test_image10.jpg': ['Flying', 'Airplane'],
    'test_image11.jpg': ['Gate', 'Airplane'],
    'test_image12.jpg': ['Terminal', 'Parking'],
    'test_image13.jpg': ['Hangar', 'Jet Bridge'],
}

# mapping labels to flight states
taxiing_labels = {'Runway', 'Airport', 'Tarmac'}
in_flight_labels = {'Flying', 'Sky', 'Airplane'}
grounded_labels = {'Gate', 'Terminal', 'Hangar', 'Parking', 'Jet Bridge'}

# function to do the classifying
def classify_flight_state(labels):
    if any(label in taxiing_labels for label in labels):
        return "Taxiing"
    elif any(label in in_flight_labels for label in labels):
        return "In-Flight"
    elif any(label in grounded_labels for label in labels):
        return "Grounded"
    else:
        return "Unknown"

# writin csv output to a file. This creates the csv file and writes the output into it. The csv file should be
#created in the same directory as the script
#if you run this script multiple times, the previous csv file will be overwritten
with open('flight_state_outputs.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Image Name', 'Flight State'])  # header

    for image_name, labels in rekognition_outputs.items():
        state = classify_flight_state(labels)
        writer.writerow([image_name, state])

print("Finished writing to flight_state_outputs.csv ")
