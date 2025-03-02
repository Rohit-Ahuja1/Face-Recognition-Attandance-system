import cv2
import face_recognition
import os
import pickle
import csv
from datetime import datetime, timedelta

# Directory for student images
img_dir = "images"
student_images = []
student_ids = []

# Get all images of students and their IDs from filenames
for img_path in os.listdir(img_dir):
    roll_num = img_path.split('.')[0]  # Assume filename format is "rollnumber.jpg"
    student_images.append(cv2.imread(os.path.join(img_dir, img_path)))
    student_ids.append(roll_num)

# Encoding generator function with error handling
def encode_images(images, ids):
    face_encodings = []
    valid_ids = []

    for idx, img in enumerate(images):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img_rgb)
        if encodings:
            face_encodings.append(encodings[0])
            valid_ids.append(ids[idx])
        else:
            print(f"No face detected in image {ids[idx]}. Skipping.")

    return face_encodings, valid_ids

# Generate and save encodings if not already saved
face_encodings, valid_ids = encode_images(student_images, student_ids)
with open("encodings.p", 'wb') as f:
    pickle.dump([face_encodings, valid_ids], f)

print("Encodings and IDs saved successfully!")

# Attendance marking function
def mark_attendance(roll_number):
    attendance_file = "attendance.csv"
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M:%S")
    
    # Check if roll number is already marked for today
    try:
        with open(attendance_file, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if row and row[0] == roll_number and row[2] == current_date:
                    print(f"Roll number {roll_number} is already marked for today.")
                    return
    except FileNotFoundError:
        pass

    # Append attendance record to CSV file
    with open(attendance_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([roll_number, current_time, current_date])
        print(f"Marked attendance for roll number {roll_number} at {current_time} on {current_date}")

# Load saved encodings for attendance marking
with open("encodings.p", 'rb') as f:
    loaded_encodings, loaded_ids = pickle.load(f)

# Initialize the webcam
cap = cv2.VideoCapture(0)
start_time = datetime.now()
max_duration = timedelta(seconds=5)  # Set max duration to 10 seconds

while True:
    # Check if max duration has passed
    if datetime.now() - start_time > max_duration:
        print("Max duration reached, closing camera.")
        break

    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image from camera.")
        break

    # Convert the captured frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Locate faces in the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    frame_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Compare each detected face encoding with saved encodings
    for frame_encoding in frame_encodings:
        matches = face_recognition.compare_faces(loaded_encodings, frame_encoding)
        if True in matches:
            match_index = matches.index(True)
            matched_roll_number = loaded_ids[match_index]
            
            # Mark attendance for the matched roll number
            mark_attendance(matched_roll_number)
        else:
            print("No match found.")

    # Display the video feed
    cv2.imshow("Attendance System", frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
