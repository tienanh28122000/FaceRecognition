import cv2
import numpy as np

s1_paths = [
    "./data/orl_faces/s1/1.pgm",
    "./data/orl_faces/s1/2.pgm",
    "./data/orl_faces/s1/3.pgm",
    "./data/orl_faces/s1/4.pgm",
    "./data/orl_faces/s1/5.pgm",
    "./data/orl_faces/s1/6.pgm",
    "./data/orl_faces/s1/7.pgm",
    "./data/orl_faces/s1/8.pgm",
    "./data/orl_faces/s1/9.pgm",
]
s2_paths = [
    "./data/orl_faces/s2/1.pgm",
    "./data/orl_faces/s2/2.pgm",
    "./data/orl_faces/s2/3.pgm",
    "./data/orl_faces/s2/4.pgm",
    "./data/orl_faces/s2/5.pgm",
    "./data/orl_faces/s2/6.pgm",
    "./data/orl_faces/s2/7.pgm",
    "./data/orl_faces/s2/8.pgm",
    "./data/orl_faces/s2/9.pgm",
]
s3_paths = [
    "./data/orl_faces/s3/1.pgm",
    "./data/orl_faces/s3/2.pgm",
    "./data/orl_faces/s3/3.pgm",
    "./data/orl_faces/s3/4.pgm",
    "./data/orl_faces/s3/5.pgm",
    "./data/orl_faces/s3/6.pgm",
    "./data/orl_faces/s3/7.pgm",
    "./data/orl_faces/s3/8.pgm",
    "./data/orl_faces/s3/9.pgm",
]
s4_paths = [
    "./data/orl_faces/s4/1.pgm",
    "./data/orl_faces/s4/2.pgm",
    "./data/orl_faces/s4/3.pgm",
    "./data/orl_faces/s4/4.pgm",
    "./data/orl_faces/s4/5.pgm",
    "./data/orl_faces/s4/6.pgm",
    "./data/orl_faces/s4/7.pgm",
    "./data/orl_faces/s4/8.pgm",
    "./data/orl_faces/s4/9.pgm",
]
s5_paths = [
    "./data/orl_faces/s5/1.pgm",
    "./data/orl_faces/s5/2.pgm",
    "./data/orl_faces/s5/3.pgm",
    "./data/orl_faces/s5/4.pgm",
    "./data/orl_faces/s5/5.pgm",
    "./data/orl_faces/s5/6.pgm",
    "./data/orl_faces/s5/7.pgm",
    "./data/orl_faces/s5/8.pgm",
    "./data/orl_faces/s5/9.pgm",
]


def average_face_vector(vector):
    m = vector.__len__()
    newVector = np.array([])

    for number in vector:
        newVector = np.concatenate((newVector, np.array([number / m])))

    return newVector


def vector_to_image(vector, image_size):

    image = []

    for i in range(image_size):
        image.append([])
        for j in range(image_size):
            image[i].append(vector[image_size * i + j])

    image = np.array(image)
    return image


def vector_normalization(images, number_of_images, image_size):

    arr = []

    for imagePath in images:
        image = cv2.imread(imagePath)
        imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        imageGrayVector = matrix_to_vector(imageGray)
        averageFaceVector = average_face_vector(imageGrayVector)
        normalizedVector = imageGrayVector - averageFaceVector
        arr.append(normalizedVector.tolist())

    arr = np.array(arr)

    return arr


def main(image_paths):

    images_vectors = np.zeros(10304, dtype=np.int8)

    vectors = []

    for image_path in image_paths:
        image = cv2.imread(image_path)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_gray = image_gray.flatten()

        vectors.append(image_gray)

        images_vectors = np.add(images_vectors, image_gray)

    avg_face_vector = np.divide(images_vectors, image_paths.__len__())

    for i in range(len(vectors)):
        vectors[i] = np.subtract(vectors[i], avg_face_vector)

    vectors = np.array(vectors)

    c_matrix = np.matmul(vectors, vectors.transpose())

    # print(c_matrix)

    eigenvalues, eigenvectors = np.linalg.eig(c_matrix)

    print(eigenvectors)
    print(max(eigenvalues))


if __name__ == "__main__":
    main(s1_paths)



# arr = vector_normalization(images_path, 10, 10304)
#
# final_matrix = np.dot(arr, arr.transpose())
#
# print(arr)
# print(arr.transpose())
#
# print(final_matrix)



# image = cv2.imread(f"{imageFolders[0]}1.pgm")
# imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# # Translate our image matrix to vector
# imageGrayVector = matrix_to_vector(imageGray)
#
# # Calculating average face vector
# averageFaceVector = average_face_vector(imageGrayVector)
#
# normalizedVector = imageGrayVector - averageFaceVector
