from ultralytics import YOLO

# Load license plate model
model = YOLO("models/best.pt")

# Test on bus image
results = model("data/car.jpg", conf=0.4)

# Show result
for result in results:
    result.show()