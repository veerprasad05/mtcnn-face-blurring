import cv2

def list_available_cameras(max_index=10):
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap is not None and cap.isOpened():
            print(f"Camera found at index {i}")
            available.append(i)
            cap.release()
        else:
            print(f"No camera at index {i}")
    return available

if __name__ == "__main__":
    available_cams = list_available_cameras()
    if not available_cams:
        print("⚠️ No available cameras found.")
    else:
        print(f"✅ Available camera indices: {available_cams}")
