import cv2
import torch
import time

RED = (0, 0, 255)
YELLOW = (0, 255, 255)
GREEN = (0, 255, 0)

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    model = torch.hub.load('ultralytics/yolov5', 'yolov5n')
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects
        results = model(frame)

        # Initialize traffic light color and text
        TL_color = GREEN
        TL_text = "Green"
        
        # Process detections(puede que "class_id" se tenga que cambiar)
        for *coords, class_id, confidence in results.xyxy[0]:
            label = f'{confidence:.2f}'
            x1, y1, x2, y2 = map(int, coords)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            #Logic
            if label == '0.00' or label == '14.00':
                
                TL_color = YELLOW
                TL_text = "Yellow"
               
        
        # Draw traffic light
        SFc= cv2.circle(frame, (30, 30), 20, TL_color, -1)
        SFt= cv2.putText(frame, TL_text, (60, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
        # Display the frame
        cv2.imshow('YOLOv5 Object Detection', frame)

        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()