import numpy as np
from keras_facenet import FaceNet
import cv2
import os
from ultralytics import YOLO

# Initialize FaceNet model
print("Loading FaceNet model...")
facenet_model = FaceNet()
print("FaceNet model loaded successfully.")

# Initialize YOLO model
print("Loading YOLO model...")
yolo_model = YOLO('myf_face_recognition/yolov8n-face.pt')
print("YOLO model loaded successfully.")

def face_recognition(cropped_image, embedding_files, threshold=0.65):
    """
    Recognize faces in the cropped image and return the recognized person along with the similarity score.

    Args:
        cropped_image (numpy.ndarray): Cropped face region from the input image.
        embedding_files (list): List of paths to the embedding files.
        threshold (float, optional): Threshold value for similarity score. Default is 0.65.

    Returns:
        tuple: Recognized person name and the maximum similarity score.
    """
    loaded_face_embeddings = [np.load(file) for file in embedding_files]

    # Find embeddings of the face in the test image
    test_face_embeddings = facenet_model.embeddings([np.array(cropped_image)])

    # Compare the test face embeddings with the loaded embeddings
    similarity_scores = [np.dot(loaded_face_embeddings[i], test_face_embeddings.T) for i in range(len(loaded_face_embeddings))]

    # Check if the test face belongs to any person in the database
    max_score = np.max(similarity_scores)
    if max_score > threshold:
        label = np.argmax(similarity_scores)
        base_name = os.path.basename(embedding_files[label])
        name_without_suffix = base_name.replace("_embeddings.npy", "")
        recognized_person = name_without_suffix.replace("_", " ").title()
    else:
        recognized_person = "Unknown"

    return recognized_person, max_score

def identify_faces_in_video(test_image, embedding_files, threshold=0.65, show_frame=True):
    """
    Identify faces in the input image using YOLO model and return recognized faces along with their similarity scores.

    Args:
        test_image (numpy.ndarray): Input image.
        embedding_files (list): List of paths to the embedding files.
        threshold (float, optional): Threshold value for similarity score. Default is 0.65.
        show_frame (bool, optional): Flag to display the processed image with recognized faces. Default is True.

    Returns:
        dict: Dictionary containing recognized faces and their similarity scores.
    """
    global facenet_model

    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    image_np_original = np.array(test_image)

    results = yolo_model.predict(test_image, show=False, stream=False)
    recognized_faces = {}  # Dictionary to store recognized faces and their similarity scores

    if results:
        for r in results:
            for box, score, class_id in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
                xmin, ymin, xmax, ymax = map(int, box.tolist())
                if show_frame:
                    cv2.rectangle(image_np_original, (xmin, ymin), (xmax, ymax), (0, 255, 0), 5)

                # Crop the face region
                face_roi = cv2.resize(test_image[ymin:ymax, xmin:xmax], (160, 160))

                # Perform face recognition using the face_recognition function
                recognized_person, max_score = face_recognition(face_roi, embedding_files, threshold=threshold)

                if recognized_person == "Unknown":
                    continue
                else:
                    recognized_faces[recognized_person] = max_score  # Add to recognized_faces dictionary

                if show_frame:
                    cv2.putText(image_np_original, f"{recognized_person} ({max_score:.2f})", (xmin, ymin - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        image_np_original = cv2.cvtColor(image_np_original, cv2.COLOR_BGR2RGB)
        if show_frame:
            cv2.imshow("Frame", image_np_original)
    else:
        print("No faces found")

    return recognized_faces  # Return dictionary containing recognized faces and their similarity scores
