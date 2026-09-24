# tests/test_yolo_image.py

from core.vision_detector import VisionDetector

def main():
    detector = VisionDetector()
    # Use any image path you have, e.g. 'data/test.jpg'
    image_path = "data/image.jpeg"
    print(f"Running detection on {image_path}...")
    results = detector.detect_image(image_path)
    print("Detection completed. Results object:", results)

if __name__ == "__main__":
    main()