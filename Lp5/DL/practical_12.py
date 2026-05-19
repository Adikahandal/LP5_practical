import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load Model
net = cv2.dnn.readNetFromCaffe(
    'colorization_deploy_v2.prototxt',
    'colorization_release_v2.caffemodel'
)

pts = np.load('pts_in_hull.npy')

# Load Cluster Centers
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")

pts = pts.transpose().reshape(2, 313, 1, 1)

net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# Load Input Image
image = cv2.imread('bw_image.jpg')

scaled = image.astype("float32") / 255.0

lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

# Resize Image
resized = cv2.resize(lab, (224, 224))

L = cv2.split(resized)[0]

L -= 50

# Predict Color
net.setInput(cv2.dnn.blobFromImage(L))

ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

# Resize to Original Size
ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

L = cv2.split(lab)[0]

# Combine Channels
colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

# Convert Back to RGB
colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)

colorized = np.clip(colorized, 0, 1)

# Show Images
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.title("Original B&W Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.subplot(1,2,2)
plt.title("Colorized Image")
plt.imshow(cv2.cvtColor((colorized * 255).astype("uint8"), cv2.COLOR_BGR2RGB))

plt.show()

# Save Output
cv2.imwrite("colorized_output.jpg", (colorized * 255).astype("uint8"))

print("Colorized image saved as colorized_output.jpg")



#  pip install opencv-python numpy matplotlib

# https://huggingface.co/spaces/BilalSardar/Black-N-White-To-Color/blob/main/colorization_release_v2.caffemodel?utm_source=chatgpt.com