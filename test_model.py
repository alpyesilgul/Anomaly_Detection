from utils import quantify_image
import cv2
import argparse
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True,
                    help="path to trained anomaly detection model")
    parser.add_argument("-i", "--image", required=True,
                    help="path to input image")
    args = vars(parser.parse_args())

    print("Loading anomaly detection model...")
    model = pickle.loads(open(args["model"], "rb").read())
    image = cv2.imread(args["image"])
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    features = quantify_image(hsv, bins=(3, 3, 3))
    pred = model.predict([features])[0]
    label = "anomaly" if pred == -1 else "normal"
    print("Close the window for another test")
    cv2.imshow(str(label), image)
    cv2.waitKey(0)

