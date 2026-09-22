from paddleocr import PaddleOCR

ocr = PaddleOCR(lang="en")

image_path = "results/plate_1.jpg"
result = ocr.predict(image_path)

for res in result:
    texts = res.get("rec_texts", [])
    scores = res.get("rec_scores", [])

    if texts and scores:
        plate = texts[0]
        confidence = scores[0] * 100

        print("Plate:", plate)
        print("Confidence:", round(confidence, 2), "%")
