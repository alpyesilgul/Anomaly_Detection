import argparse
from anomaly_detection import model
import pickle


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--images_dir", required=True,
	help="path to images_dir")
	parser.add_argument("-m", "--model", required=True,
	help="path to output anomaly detection model")

	args = vars(parser.parse_args())
	model = model(args['images_dir'])
	try:
		f = open(args['model'], "wb")
		f.write(pickle.dumps(model))
		f.close()
	except PermissionError:
		print('path must be PATH+model_name.model')
