import cv2

# Load the vehicle cascade using your local path
vehicle_cascade = cv2.CascadeClassifier(r'C:\Users\sharukesh\Downloads\haarcascade_car.xml')

# Check if the cascade is loaded correctly
if vehicle_cascade.empty():
    print("Error: Cascade classifier not loaded properly.")
    exit()

# Start video capture (from file or camera)
cap = cv2.VideoCapture(r'C:\Users\sharukesh\Downloads\real-time.mp4')

# Check if the video capture opened correctly
if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

# Initialize the vehicle count
vehicle_count = 0

# Start the loop to process frames
while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Empty frame received.")
        break  # This break is now correctly inside the while loop
    
    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect vehicles in the frame (tuning parameters)
    vehicles = vehicle_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # If no vehicles detected, continue to the next frame
    if len(vehicles) == 0:
        continue

    # Draw rectangles around detected vehicles and update the vehicle count
    for (x, y, w, h) in vehicles:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        vehicle_count += 1  # Increase count for each detected vehicle
    
    # Add vehicle count text in the upper-right corner of the frame
    cv2.putText(frame, f"Vehicles: {vehicle_count}", 
                (frame.shape[1] - 200, 50),  # Position (top-right corner)
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Display the processed frame
    cv2.imshow('Vehicle Detection', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()