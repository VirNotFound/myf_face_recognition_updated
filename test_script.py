from myf_face_recognition.embeddings import generate_face_embeddings
from myf_face_recognition.images import identify_faces_in_image
from myf_face_recognition.video import identify_faces_in_video
from myf_face_recognition.simple_video import simple_identify_faces_in_video

from myf_face_recognition.all import generate_face_embeddings, identify_faces_in_image, identify_faces_in_video, simple_identify_faces_in_video

import cv2

# Set a threshold for similarity
threshold = 0.64

# List of embedding files
embedding_files = [
    "embeddings/Tony_Stark_embeddings.npy",
    # Add more paths as needed
]



# Example for generating face embeddings

name = "Tony Stark"
folder_path = 'Tony Stark'
output_folder_path = 'embeddings'
generate_face_embeddings(name, folder_path, output_folder_path, show_training=True)



# Example for image recognition

test_image = cv2.imread("test_tony.jpg")
recognized_faces= identify_faces_in_image(test_image, embedding_files, show_frame=True)
print(recognized_faces)



# Example for video recognition

video_path = "Tony.mp4"
cap = cv2.VideoCapture(video_path)
while True:
    ret, frame = cap.read()
    if not ret:
        print("Couldn't Load")
        break

    results = identify_faces_in_video(frame, embedding_files)
    print(results)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

cv2.destroyAllWindows()
