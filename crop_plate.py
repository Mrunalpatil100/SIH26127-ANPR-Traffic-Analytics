from ultralytics import YOLO
import cv2
import os

# Load license plate model
model = YOLO("models/best.pt")

# Read car image
image_path = "data/car.jpg"
image = cv2.imread(image_path)

# Detect license plate
results = model(image, conf=0.4)

# Create results folder if it doesn't exist
os.makedirs("results", exist_ok=True)

# Crop and save detected plates
count = 0

for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        plate = image[y1:y2, x1:x2]

        count += 1
        output_path = f"results/plate_{count}.jpg"

        cv2.imwrite(output_path, plate)

        print(f"License plate saved: {output_path}")

print(f"Total plates detected: {count}")