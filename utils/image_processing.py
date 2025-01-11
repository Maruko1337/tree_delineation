import cv2

def image_pyramid(img, scales=[1.0, 0.5, 0.25]):
    pyramid_images = []
    for scale in scales:
        resized_img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        pyramid_images.append(resized_img)
    return pyramid_images
