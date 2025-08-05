📄 OCR & Real-Time Text Detection
This project uses OpenCV's EAST text detector and Tesseract OCR to detect and recognize text in real time from images, webcam, or video files.

✅ Requirements
OpenCV 3.4.2 or above
Tesseract 4.0 or above

📝 Project Description
This project implements Real-Time Optical Character Recognition (OCR) using a combination of OpenCV’s EAST (Efficient and Accurate Scene Text Detector) and Tesseract OCR. It is designed to detect and recognize text from various sources, including static images, video files, and live webcam feeds.

The system works in two main stages:

Text Detection: Using the EAST deep learning model provided by OpenCV, it accurately identifies regions in an image or frame that contain text. This method works on both printed and scene text with good performance in real-time.

Text Recognition: Once text regions are detected, the cropped regions are passed to Tesseract OCR, which extracts readable text content from them.

This pipeline allows you to visually capture, extract, and display text from real-world visual inputs like documents, signboards, number plates, and screens — all in real-time.

The project is modular, with separate scripts for handling:

Image input

Webcam feed

Video files

Standalone detection or full recognition

It is useful for building OCR-based applications such as:

Assistive technology (for visually impaired users)

Smart document scanners

Text-based search from video

Real-time translation tools

License plate recognition systems

The project is written in Python and requires a compatible version of OpenCV (3.4.2+) and Tesseract (4.0+).

