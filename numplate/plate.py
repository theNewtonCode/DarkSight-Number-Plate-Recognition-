import cv2
import matplotlib.pyplot as plt

class LicensePlateSegmenter:
    def __init__(self, image_path):
        self.image_path = image_path
        self.plate_cascade = cv2.CascadeClassifier('./indian_license_plate.xml')
        
    def detect_plate(self, img, text=''): # the function detects and performs blurring on the number plate.
        plate_img = img.copy()
        roi = img.copy()
        plate_rect = self.plate_cascade.detectMultiScale(plate_img, scaleFactor = 1.2, minNeighbors = 7) # detects numberplates and returns the coordinates and dimensions of detected license plate's contours.
        for (x,y,w,h) in plate_rect:
            roi_ = roi[y:y+h, x:x+w, :] # extracting the Region of Interest of license plate for blurring.
            plate = roi[y:y+h, x:x+w, :]
            cv2.rectangle(plate_img, (x+2,y), (x+w-3, y+h-5), (51,181,155), 3) # finally representing the detected contours by drawing rectangles around the edges.
        if text!='':
            plate_img = cv2.putText(plate_img, text, (x-w//2,y-h//2), 
                                    cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.5, (51,181,155), 1, cv2.LINE_AA)

        return plate_img, plate # returning the processed image.
    
    def segment_characters(self):
        img = cv2.imread(self.image_path)
        
        # Detect license plate and extract plate region
        _, plate = self.detect_plate(img)
        
        # Preprocess cropped license plate image
        img_lp = cv2.resize(plate, (333, 75))
        img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
        _, img_binary_lp = cv2.threshold(img_gray_lp, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        img_binary_lp = cv2.erode(img_binary_lp, (3,3))
        img_binary_lp = cv2.dilate(img_binary_lp, (3,3))

        LP_WIDTH = img_binary_lp.shape[0]
        LP_HEIGHT = img_binary_lp.shape[1]

        # Make borders white
        img_binary_lp[0:3,:] = 255
        img_binary_lp[:,0:3] = 255
        img_binary_lp[72:75,:] = 255
        img_binary_lp[:,330:333] = 255

        # Estimations of character contours sizes of cropped license plates
        dimensions = [LP_WIDTH/6,
                       LP_WIDTH/2,
                       LP_HEIGHT/10,
                       2*LP_HEIGHT/3]

        # Save processed image to file and return file path
        file_path = 'contour.jpg'
        cv2.imwrite(file_path, img_binary_lp)
        return file_path

    def display(self, img_, title=''):
        img = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
        fig = plt.figure(figsize=(10,6))
        ax = plt.subplot(111)
        ax.imshow(img)
        plt.axis('off')
        plt.title(title)
        plt.show()