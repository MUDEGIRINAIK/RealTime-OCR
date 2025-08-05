from imutils.object_detection import non_max_suppression
import numpy as np
import cv2
from utils import forward_passer, box_extractor

def resize_image(image, width, height):
    """
    Re-sizes image to given width & height
    """
    h, w = image.shape[:2]
    ratio_w = w / width
    ratio_h = h / height
    image = cv2.resize(image, (width, height))
    return image, ratio_w, ratio_h

def main():
    image_path = "test.png"  # Set the image path here
    east_model_path = "frozen_east_text_detection.pb"  # Set EAST model path
    min_confidence = 0.5
    width, height = 320, 320

    # reading in image
    image = cv2.imread(r"C:\Users\girin\OneDrive\Documents\Q technologies\Project6\real-time-OCR\image.png")

    if image is None:
        print("Error: Unable to load image. Check the file path.")
        return
    
    orig_image = image.copy()

    # resizing image
    image, ratio_w, ratio_h = resize_image(image, width, height)

    # layers used for ROI recognition
    layer_names = ['feature_fusion/Conv_7/Sigmoid', 'feature_fusion/concat_3']

    # pre-loading the frozen graph
    print("[INFO] loading the detector...")
    net = cv2.dnn.readNet(east_model_path)

    # getting results from the model
    scores, geometry = forward_passer(net, image, layers=layer_names)

    # decoding results from the model
    rectangles, confidences = box_extractor(scores, geometry, min_confidence)

    # applying non-max suppression to get boxes depicting text regions
    boxes = non_max_suppression(np.array(rectangles), probs=confidences)

    # drawing rectangles on the image
    for (start_x, start_y, end_x, end_y) in boxes:
        start_x = int(start_x * ratio_w)
        start_y = int(start_y * ratio_h)
        end_x = int(end_x * ratio_w)
        end_y = int(end_y * ratio_h)
        cv2.rectangle(orig_image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

    cv2.imshow("Detection", orig_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
