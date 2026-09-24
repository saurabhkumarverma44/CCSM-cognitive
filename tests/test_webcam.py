# tests/test_webcam.py

import cv2
from core.camera import Camera

def main():
    cam = Camera(device_index=0)  # 0 is usually the default laptop webcam
    cam.open()
    print("Press 'q' to quit.")
    while True:
        frame = cam.read_frame()
        cv2.imshow("CCSM Webcam Test", frame)
        # Wait 1 ms and check for 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()