import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Read the image
img_bgr = cv2.imread("image.jpg")        # put your image file here
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # convert BGR -> RGB for plt

# 2. Apply a BLUR (low-pass filter: removes noise)
blur = cv2.GaussianBlur(img_rgb, (7, 7), 0)

blur1 = cv2.GaussianBlur(img_rgb, (3, 3), 0)
blur2 = cv2.GaussianBlur(img_rgb, (7, 7), 0)
blur3 = cv2.GaussianBlur(img_rgb, (15, 15), 0)

# 3. Apply a SHARPEN filter (high-pass-ish)
kernel_sharpen = np.array([[0, -1,  0],
                           [-1, 5, -1],
                           [0, -1,  0]], dtype=np.float32)

sharpen = cv2.filter2D(img_rgb, -1, kernel_sharpen)

# 4. Apply EDGE DETECTION (detect boundaries)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)    # Canny needs grayscale
edges = cv2.Canny(img_gray, 100, 200)



# 5. Show all results
plt.figure(figsize=(12, 8))

plt.subplot(3, 3, 1)
plt.title("Original")
plt.imshow(img_rgb)
plt.axis("off")

plt.subplot(3, 3, 2)
plt.title("Blurred (Low-pass)")
plt.imshow(blur)
plt.axis("off")

plt.subplot(3, 3, 3)
plt.title("Sharpened")
plt.imshow(sharpen)
plt.axis("off")

plt.subplot(3, 3, 4)
plt.title("Edges (Canny)")
plt.imshow(edges, cmap="gray")
plt.axis("off")


plt.subplot(3, 3, 5)
plt.title("Blurred (Low-pass)")
plt.imshow(blur1)
plt.axis("off")


plt.subplot(3, 3, 6)
plt.title("Blurred (Low-pass)")
plt.imshow(blur2)
plt.axis("off")


plt.subplot(3, 3, 7)
plt.title("Blurred (Low-pass)")
plt.imshow(blur3)
plt.axis("off")


plt.tight_layout()
plt.show()



