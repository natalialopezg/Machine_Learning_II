# ==============================================================================
#                               LIBRARY IMPORT
# ==============================================================================
# Multidimensional array object
import numpy as np

# Graphics
import matplotlib.pyplot as plt
from PIL import Image

# Operating System tasks
import os

# ==============================================================================
#                          WORKSHOP SOLUTION - POINT 2
# ==============================================================================

# 2) Add a steady, well-centered picture of your face to a shared folder 
# alongside your classmates
# ==============================================================================
# Read original picture
original_picture= Image.open(r'LAB1\Original_picture.jpg')

# Grayscale transformation
gray_image = original_picture.convert('L')

# Resize to 256x256 pixels
w,h  = 256, 256
modified_image = gray_image.resize((w,h))

# Save modified picture
modified_image.save('LAB1\Modified_picture.jpg')

# Plot edited face
fig = plt.figure(1,figsize=(10, 5)) 
fig.add_subplot(1, 2, 1) 
plt.imshow(original_picture) 
plt.title("Original picture")
fig.add_subplot(1, 2, 2) 
plt.imshow(modified_image, cmap='gray') 
plt.title("Modified picture")
# plt.show()

# Calculate and plot the average face of the cohort
# ==============================================================================
# Read files in directory
files = os.listdir('LAB1\pictures')
filenames = [filename for filename in files if filename[-4:].lower() in [".jpg",".png"]]

# Create array for result
result = np.zeros((w,h, 1),float)

# Load pictures
pictures = []
for i,picture_name in enumerate(filenames):
    # picture_name = "Natalia_Lopez.jpg" # Test distance when average face is just my face.
    path = f"LAB1\pictures\{picture_name}"
    picture = Image.open(path).convert('L')
    picture_array = np.array(picture,dtype= float)
    if (picture_array.shape[0] != w) | (picture_array.shape[1] != h):
        picture = picture.resize((w,h))
        # print(f"{picture_name} resized")

    pictures.insert(i,picture)

# Array of picture arrays
pictures_array = np.array([np.array(image,dtype= float) for image in pictures])

# Average calculation
average_array = np.average(pictures_array,axis=0)

# Create image from array
average_face = Image.fromarray(average_array.astype('uint8'),mode="L")

# Plot average face
fig2 = plt.figure(2,figsize=(5, 5)) 
plt.imshow(average_face, cmap='gray')
plt.title('Average face')
plt.axis('off')

# Distance from average
my_face = Image.open(r"LAB1\pictures\Natalia_Lopez.jpg").convert('L')
my_face_array = np.array(my_face,dtype= float)
avg_array = np.array(average_array,dtype= float)

diff =  avg_array - my_face_array

distance = np.sqrt(np.sum(np.square(diff)))
print(f"My face is {distance:,.0f} pixels distant from average face of the cohort\n")
print(f"To measure de distance between my face and de average face of the cohort \
I use the Euclidean Distance to measure the matrix distance. This distance corresponds to \
the square root of the squared differences between corresponding elements.\n")

plt.show()
