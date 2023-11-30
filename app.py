
def get_correspondences(frame1,book_img,num_correspondences=50):
    
        sift = cv2.SIFT_create()

        kp1, des1 = sift.detectAndCompute(book_img, None)
        kp2, des2 = sift.detectAndCompute(frame1, None)

        bf = cv2.BFMatcher()

        matches = bf.knnMatch(des1, des2, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.25 * n.distance:
                good_matches.append(m)

        good_matches = good_matches[:num_correspondences]

        #save the correspondnces 
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
       
        return pts1, pts2




def map_points_with_homography(H, points):
        points_homogeneous = np.hstack((points, np.ones((len(points), 1))))
        mapped_points_homogeneous = np.dot(H, points_homogeneous.T)
        # Normalize homogeneous coordinates
        mapped_points = mapped_points_homogeneous[:2, :] / mapped_points_homogeneous[2, :].reshape(1, -1)

        return mapped_points.T
def compute_homography(pts1, pts2):
    A = []
    for i in range(0, len(pts1)):
        x, y = pts1[i][0], pts1[i][1]
        u, v = pts2[i][0], pts2[i][1]
        A.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
        A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])

    A = np.asarray(A)
    U, S, Vh = np.linalg.svd(A)
    H = Vh[-1, :].reshape(3, 3)
    # Normalize H (optional)
    H /= H[2, 2]
    return H
def RANSAC(pts1, pts2, num_iterations=1000, min_set_size=4, inlier_threshold=.5, min_inliers=45):
    best_H = None
    max_inliers = 0
    num_correspondences = len(pts1)
    np.random.seed(42)
    for i in range(num_iterations):
        random_indices = np.random.choice(num_correspondences, size=min_set_size, replace=False)
        sampled_pts1 = pts1[random_indices]
        sampled_pts2 = pts2[random_indices]

        initial_H = compute_homography(sampled_pts1, sampled_pts2)
        transformed_points = map_points_with_homography(initial_H, pts1)

        errors = np.sqrt(np.sum((transformed_points - pts2)**2, axis=1))
        inliers = np.sum(errors < inlier_threshold)

        if inliers > max_inliers:
            max_inliers = inliers
            best_H = initial_H

        if max_inliers > min_inliers:
            break

    return best_H



def calculate_book_coordinates(H,book_corners):
    mapped_corners_homogeneous = map_points_with_homography(H, (book_corners))
    return mapped_corners_homogeneous

def crop_ar_video_frame(video_frame, book_corners, cut_edges=True, y_edge_width=22, x_edge_width = 0):
    
    width = book_corners[1][0] - book_corners[0][0]
    height = book_corners[3][1] - book_corners[0][1]
    
    start_y = math.ceil((video_frame.shape[0] - height)/2)
    end_y = video_frame.shape[0] - start_y
    start_x = math.ceil((video_frame.shape[1] - width)/2)
    end_x = video_frame.shape[1] - start_x
    if cut_edges:
        cropped_frame = video_frame[start_y+y_edge_width:end_y-y_edge_width, start_x+x_edge_width:end_x-x_edge_width]
    else:
        cropped_frame = video_frame[start_y:end_y, start_x:end_x]
        
    return cv2.resize(cropped_frame, (int(width), int(height))) 






def overlay_frames(frame1, frame2, H,book_corners):
    book_coordinates_video = calculate_book_coordinates(H,book_corners)

    mask = np.zeros_like(frame1, dtype=np.uint8)
    cv2.fillPoly(mask, [np.int32(book_coordinates_video)], (255, 255, 255))

    inverted_mask = cv2.bitwise_not(mask)

    frame1_blacked = cv2.bitwise_and(frame1, inverted_mask)

    overlay_frame = cv2.warpPerspective(frame2, H, (frame1.shape[1], frame1.shape[0]))

    result = cv2.add(frame1_blacked, overlay_frame)

    # Plot book_coordinates_video on the new frame
    for i in range(len(book_coordinates_video)):
        pt1 = (int(book_coordinates_video[i, 0]), int(book_coordinates_video[i, 1]))
        cv2.circle(result, pt1, 5, (0, 255, 0), -1)  # Draw green circles at each mapped coordinate

    return result

from flask import Flask, render_template, request, redirect, url_for, send_file
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import sys
import os
import tempfile

app = Flask(__name__)

def process_video(video1_path, video2_path, book_img_path):
    # Open the video files
    video1 = cv2.VideoCapture(video1_path)
    video2 = cv2.VideoCapture(video2_path)

    # Load the book image
    book_img = cv2.imread(book_img_path)
    if book_img is None:
        raise ValueError("Image not found at path: " + book_img_path)

    # Your existing code for setting up the video writer
    width = int(video1.get(3))
    height = int(video1.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_path = os.path.join(tempfile.gettempdir(), 'output_video.avi')
    output_video = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

    book_corners = np.array([[0, 0],
                             [book_img.shape[1] - 1, 0],
                             [book_img.shape[1] - 1, (book_img.shape[0] - 1)],
                             [0, (book_img.shape[0] - 1)]],
                            dtype=np.float32)

    # Main processing loop
    while True:
        ret1, frame1 = video1.read()
        ret2, frame2 = video2.read()

        if not ret1 or not ret2:
            break

        # Get correspondences for each frame
        pts_book, pts_video = get_correspondences(frame1,book_img)
        # Calculate homography matrix
        H = RANSAC(np.squeeze(pts_book), np.squeeze(pts_video))

        # Calculate book coordinates for the overlay
        book_coordinates = calculate_book_coordinates(H, book_corners)

        # Crop the video frame centered on the book
        cropped_video_frame = crop_ar_video_frame(frame2, book_corners)

        # Overlay frames and write to the output video
        result_frame = overlay_frames(frame1, cropped_video_frame, H,book_corners)
        output_video.write(result_frame)

    # Release video captures and writer
    video1.release()
    video2.release()
    output_video.release()

    return output_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    if request.method == 'POST':
        video1_path = request.files['video1'].read()
        video2_path = request.files['video2'].read()
        book_img_path = request.files['book_img'].read()

        with tempfile.NamedTemporaryFile(delete=False, suffix='.mov') as video1_tempfile:
            video1_tempfile.write(video1_path)
            video1_tempfile_path = video1_tempfile.name

        with tempfile.NamedTemporaryFile(delete=False, suffix='.mov') as video2_tempfile:
            video2_tempfile.write(video2_path)
            video2_tempfile_path = video2_tempfile.name

        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as book_img_tempfile:
            book_img_tempfile.write(book_img_path)
            book_img_tempfile_path = book_img_tempfile.name

        output_path = process_video(video1_tempfile_path, video2_tempfile_path, book_img_tempfile_path)

        return send_file(output_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
