import cv2

# Function to detect faces, draw bounding boxes, and predict age/gender
def faceBoxAndPredict(faceNet, ageNet, genderNet, frame):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], swapRB=False)
    faceNet.setInput(blob)
    detection = faceNet.forward()
    bboxs = []
    
    for i in range(detection.shape[2]):
        confidence = detection[0, 0, i, 2]
        if confidence > 0.7:
            x1 = int(detection[0, 0, i, 3] * frameWidth)
            y1 = int(detection[0, 0, i, 4] * frameHeight)
            x2 = int(detection[0, 0, i, 5] * frameWidth)
            y2 = int(detection[0, 0, i, 6] * frameHeight)
            bboxs.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

            # Extract face ROI for age and gender prediction
            faceROI = frame[y1:y2, x1:x2]
            blob = cv2.dnn.blobFromImage(faceROI, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = "Male" if genderPreds[0, 0] > 0.5 else "Female"

            ageNet.setInput(blob)
            agePreds = ageNet.forward()

            # Calculate predicted age by weighted sum of age probabilities
            ageSum = 0
            for j in range(agePreds.shape[1]):
                ageSum += j * agePreds[0, j]

            # Convert ageSum to integer age (approximation)
            age = int(ageSum)

            # Display age and gender information
            label = f"{gender}, Age: {age}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame, bboxs

# Define file paths for models
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

# Load face detection model
faceNet = cv2.dnn.readNet(faceModel, faceProto)

# Load age and gender prediction models
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Open video capture
video = cv2.VideoCapture(0)

# Check if video is opened successfully
if not video.isOpened():
    print("Error: Failed to open video capture")
    exit()

# Main loop to process video frames
while True:
    # Read frame from video
    ret, frame = video.read()

    if not ret:
        print("Error: Failed to read frame from video")
        break

    # Process frame (face detection, age/gender prediction, and drawing bounding boxes)
    frame, bboxs = faceBoxAndPredict(faceNet, ageNet, genderNet, frame)

    # Display processed frame
    cv2.imshow("Face Detection with Age and Gender", frame)

    # Check for user input to exit
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Release video capture and close all windows
video.release()
cv2.destroyAllWindows()
