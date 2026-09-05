from ultralytics import YOLO
from paddleocr import PaddleOCR
import cv2
import os
import glob 

# Load models
model = YOLO("models/best.pt")
ocr = PaddleOCR(lang="en")

# Create results folder
os.makedirs("results", exist_ok=True)

# Get all images from data folder
image_files = glob.glob("data/*.jpg") + glob.glob("data/*.jpeg") + glob.glob("data/*.png")

# Process every image
for image_path in image_files:

    print("\nProcessing:", image_path)

    # Read image
    image = cv2.imread(image_path)

    # Detect license plate
    results = model(image, conf=0.4)

    count = 0

    for result in results:
        for box in result.boxes:

            # Get plate coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Crop plate
            plate = image[y1:y2, x1:x2]

            count += 1
            output_path = f"results/plate_{count}.jpg"

            # Save cropped plate
            cv2.imwrite(output_path, plate)

            print(f"License plate saved: {output_path}")

            # OCR
            ocr_result = ocr.predict(output_path)

            for res in ocr_result:
                if len(res["rec_texts"]) > 0:
                    print("Plate:", res["rec_texts"][0])
                    print(
                        "Confidence:",
                        round(res["rec_scores"][0] * 100, 2),
                        "%"
                    )

    print(f"Total plates detected: {count}")

print("\nFinished processing all images!")