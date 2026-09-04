from paddleocr import PaddleOCR

ocr = PaddleOCR(lang="en")

result = ocr.predict("results/plate_1.jpg")

for res in result:
    print("Plate:", res["rec_texts"][0])
    print("Confidence:", round(res["rec_scores"][0] * 100, 2), "%")