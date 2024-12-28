import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Path to images folder
path = 'C:/Users/amans/Desktop/vvk_Project'

images = []
classNames = []
branches = []
semesters = []
myList = os.listdir(path)
print(f"Found files: {myList}")

# Load images, names, branches, and semesters
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    # Extract name, branch, and semester from the filename (assuming "Name_Branch_Semester.jpg")
    name_branch_sem = os.path.splitext(cl)[0].split('_')
    name = name_branch_sem[0]
    branch = name_branch_sem[1] if len(name_branch_sem) > 1 else "Unknown"
    semester = name_branch_sem[2] if len(name_branch_sem) > 2 else "Unknown"
    classNames.append(name)
    branches.append(branch)
    semesters.append(semester)
print(f"Class Names: {classNames}")
print(f"Branches: {branches}")
print(f"Semesters: {semesters}")

def findEncodings(images):
    """Encodes faces from the provided images."""
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img)
        if encodings:
            encodeList.append(encodings[0])  # Add the first encoding
        else:
            print("No face detected in an image. Skipping...")
    return encodeList

def markAttendance(name, branch, semester):
    """Marks attendance for a recognized face."""
    filename = "Attendance.csv"

    # Create the file if it doesn't exist
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            f.write("Name,Branch,Semester,Time\n")  # Write header

    # Check and append attendance
    with open(filename, 'r+') as f:
        myDataList = f.readlines()
        nameList = [line.split(',')[0] for line in myDataList]

        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%Y-%m-%d %H:%M:%S')
            f.writelines(f'{name},{branch},{semester},{dtString}\n')
            print(f"Attendance marked for {name} ({branch}, {semester}) at {dtString}")

# Encode known faces
encodeListKnown = findEncodings(images)
print('Encoding Complete')

# Access the webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture frame from webcam. Exiting...")
        break

    # Resize and process the frame
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Detect faces and their encodings
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            branch = branches[matchIndex]
            semester = semesters[matchIndex]
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name, branch, semester)

    cv2.imshow('Webcam', img)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
