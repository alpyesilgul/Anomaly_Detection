## Anomaly_Detection
**using Isolation Forest with histograms for the detection task**

Since the model histograms are used, it can be used for classification of objects on the basis of color.
The model can be trained with specified instructions using command lines.
For example, there is a model trained with forest images in the example folder. Type this line in command line :
```
cd .../Anomaly_Detection/example/
```
and than

```
python test_model.py -m detection.model -i image1.jpg
```
Then the window that opens and the window should have a prediction written above it.
