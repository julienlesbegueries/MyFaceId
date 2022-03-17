import face_recognition
import cv2
import numpy as np
from os import walk

from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import imutils
import time
from playsound import playsound


app = Flask(__name__)

@app.route("/")
def index():
	# return the rendered template
	return render_template("index.html")


filenames = next(walk('.'), (None, None, []))[2]  # [] if no file
# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.



# Load a sample picture and learn how to recognize it.
#obama_image = face_recognition.load_image_file("obama.jpeg")
#obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

# Load a second sample picture and learn how to recognize it.
#biden_image = face_recognition.load_image_file("biden.jpeg")
#biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

known_face_encodings = []
known_face_names = []
for filename in filenames:
    if ".jpeg" in filename or ".jpg" in filename:
        print(filename)
        dot_index = filename.index('.')
        image = face_recognition.load_image_file(filename)
        face_encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(face_encoding)
        known_face_names.append(filename[0:dot_index])


# Create arrays of known face encodings and their names
#known_face_encodings = [
#    obama_face_encoding,
#    biden_face_encoding
#]
#known_face_names = [
#    "Barack Obama",
#    "Joe Biden"
#]

# Initialize some variables


def generate():
    # Get a reference to webcam #0 (the default one)
    video_capture = cv2.VideoCapture(0)
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    cpt_alarm = 0

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    cpt_alarm = 0
                    name = known_face_names[best_match_index]
                else:
                    cpt_alarm += 1
                    if cpt_alarm > 5:
                        playsound('alarme.wav')
                        cpt_alarm = 0
                        

            
                face_names.append(name)

        process_this_frame = not process_this_frame


        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        (flag, encodedImage) = cv2.imencode(".jpg", frame)

        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encodedImage) + b'\r\n')

        # ==== client lourd ===
        # Display the resulting image
        #cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break

        
        



@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")


# Release handle to the webcam
#video_capture.release()
#cv2.destroyAllWindows()