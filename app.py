import streamlit as st
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageFilter
from PIL import ImageEnhance
st.title('Image Processing with streamlit')


def main():
    menu = ['Home', 'Photo Restoration', 'Image Filtering', 'Face Detection',
            'Object Detection', 'Feature Detection', 'Video Processing']
    choice = st.sidebar.selectbox('Menu', menu)
    if choice == 'Home':
        home()
    if choice == 'Photo Restoration':
        photo()
    if choice == 'Image Filtering':
        images()
    if choice == 'Face Detection':
        face()
    if choice == 'Object Detection':
        objects()
    if choice == 'Feature Detection':
        features()
    if choice == 'Video Processing':
        video()


def home():
    st.title('Home')
    st.image('2462340.jpg', width=550)


def images():
    st.title('Applying Effects on the image')

    st.subheader('Choose option among the following:')
    option = st.selectbox(
        'Options', ['Threshold', 'Black and white', 'Contour', 'Emboss', 'Contrast', 'Canny Edge Detector'])
    if option == 'Threshold':
        if st.button('Normal Image'):
            normal = Image.open('b.jpg')
            st.image(normal, use_column_width=True)
        image = cv2.imread('b.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        x = st.slider('Changing threshold value', min_value=50, max_value=255)
        ret, thresh = cv2.threshold(image, x, 255, cv2.THRESH_BINARY)
        thresh = thresh.astype(np.float64)
        st.image(thresh, use_column_width=True, clamp=True)
    if option == 'Black and white':
        image = Image.open("b.jpg")
        fig = plt.figure()
        image = image.convert("1")
        plt.imshow(image)
        plt.axis('off')
        st.pyplot(fig)
    if option == 'Contour':
        image = Image.open('b.jpg')
        fig = plt.figure()
        con = image.filter(ImageFilter.CONTOUR)
        plt.imshow(con)
        plt.axis('off')
        st.pyplot(fig)
    if option == 'Emboss':
        image = Image.open('b.jpg')
        fig = plt.figure()
        con = image.filter(ImageFilter.EMBOSS)
        plt.imshow(con)
        plt.axis('off')
        st.pyplot(fig)
    if option == 'Contrast':
        image = Image.open('b.jpg')
        fig = plt.figure()
        contrast = ImageEnhance.Contrast(image).enhance(12)
        plt.imshow(contrast)
        plt.axis('off')
        st.pyplot(fig)
    if option == 'Canny Edge Detector':
        img = cv2.imread('b.jpg')
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
        edges = cv2.Canny(img_blur, 100, 200)
        cv2.imwrite('edges.jpg', edges)
        st.images(edges, use_column_width=True, clamp=True)


def face():
    st.header('Face detection')
    if st.button('Normal Image'):
        normal = Image.open('b.jpg')
        st.image(normal, use_column_width=True)
    img2 = cv2.imread('b.jpg')
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face = face_cascade.detectMultiScale(img2)
    print(f"{len(face)} faces detected in the image.")
    for x, y, width, height in face:
        cv2.rectangle(img2, (x, y), (x+width, y+height),
                      color=(255, 0, 0), thickness=2)
    cv2.imwrite('faces.jpg', img2)
    st.image(img2, use_column_width=True, clamp=True)


def objects():
    st.title('Object Detection')
    st.text("Detecting eyes from an image")
    img = cv2.imread('b.jpg')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    eye = cv2.CascadeClassifier('haarcascade_eye.xml')
    eyes = eye.detectMultiScale(img_gray, minSize=(20, 20))
    amount_found_ = len(eyes)

    if amount_found_ != 0:
        for (x, y, width, height) in eyes:

            cv2.rectangle(img_rgb, (x, y),
                          (x + height, y + width),
                          (0, 255, 0), 5)
        st.image(img_rgb, use_column_width=True, clamp=True)


def features():
    st.title('Feature Detection')
    image = cv2.imread('b.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fast = cv2.FastFeatureDetector_create()
    keypoints = fast.detect(gray, None)
    st.write('No. of keypoints detected', len(keypoints))
    image_ = cv2.drawKeypoints(
        image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    st.image(image_, use_column_width=True, clamp=True)


def photo():
    st.title('Photo Restoration')
    damaged = cv2.imread('c.jpg')
    st.write('damaged photo')
    st.image(damaged, width=400)
    ret, thresh1 = cv2.threshold(damaged, 254, 255, cv2.THRESH_BINARY)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.dilate(thresh1, kernel, iterations=1)
    restored = cv2.inpaint(damaged, mask, 3, cv2.INPAINT_NS)
    st.write('Restored image')
    st.image(restored, width=400)


def video():
    st.title('Video Restoration')


if __name__ == '__main__':
    main()
