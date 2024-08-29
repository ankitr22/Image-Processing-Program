import os  # OS interaction (e.g., file handling)
import numpy as np  # Numerical operations (arrays, matrices)
import cv2 as cv  # Image processing
import matplotlib.pyplot as plt  # Plotting and visualization


# Path to the Haar Cascade XML file for face detection
path = "C:\\Users\\Ankit\\OneDrive\\Desktop\\PlacementNotes\\haarcascade_frontalface_default.xml"

# Load the face detection algorithm from the specified XML file
classifier = cv.CascadeClassifier(path)


def user_guide():
    # Function to display user instructions
    
    print("To capture image, press: c")
    print("To exit, press: x")
    print("." * 30)


def save_image(frame, folder, image_name):
    # Function to save the captured image
    
    if not os.path.exists(folder):  # Check if the folder exists
        os.makedirs(folder)  # Create the folder if it doesn't exist

    folder_length = len(os.listdir(folder)) + 1  # Determine the new image's index
    image_path = folder + "/" + image_name + str(folder_length) + ".png"  # Construct the full path for the new image

    cv.imwrite(image_path, frame)  # Save the image to the specified path


def take_selfie():
    # Function to capture a selfie using the webcam
    
    user_guide()  # Display user instructions
    cam = cv.VideoCapture(0)  # Initialize webcam capture
    
    try:
        while True:
            ret, img = cam.read()  # Capture a frame from the webcam
            if not ret:  # Check if frame capture was successful
                print("Failed to grab frame")
                break

            image = cv.flip(img, 1)  # Flip the image horizontally for a mirror effect
            cv.imshow("Camera", image)  # Display the captured image
            
            key = cv.waitKey(20)  # Wait for a key press with a 20ms delay
            if key == ord('x'):  # Exit the loop if 'x' is pressed
                break
            
            if key == ord('c'):  # Capture and save the image if 'c' is pressed
                save_image(image, "MyPictures", "Selfie_")
                print("Selfie saved")
    finally:
        # Always release the camera and close all OpenCV windows
        cam.release()
        cv.destroyAllWindows()


def color_filter(color):
    # Create a color filter frame with a cool (blue) tone
    
    filter_frame = [] # Create coll filter frame    
    for i in range(480):  # Iterate over rows
        temp = []
        for j in range(640):  # Iterate over columns
            temp.append(color)  # Append the color value
            
        filter_frame.append(temp)  # Append the row to the frame

    filter_frame = np.array(filter_frame).astype(np.uint8)  # Convert to a NumPy array
    return filter_frame


def theme_filter():
    # Load and resize a theme image for filtering
    
    theme_path = "theme1.jpg"   # theme image path
    
    theme_frame = cv.imread(theme_path)  # Read the theme image
    if len(theme_frame.shape) == 2:  # Convert grayscale image to BGR if necessary
        theme_frame = cv.cvtColor(theme_frame, cv.COLOR_GRAY2BGR)
    
    theme_frame = cv.resize(theme_frame, (640, 480))  # Resize the theme to match the webcam frame size
    return theme_frame



def filter_selfie(color, filter_type):
    # Function to apply filter using filter-frame on webcam feed
    
    cam = cv.VideoCapture(0)  # Initialize webcam capture
    if filter_type == "color":
        filter_frame = color_filter(color)  # Apply color filter
    elif filter_type == "theme":
        filter_frame = theme_filter()  # Apply theme filter
    else:
        print("Invalid filter type!")  # Handle invalid filter types
        return
    
    try:
        user_guide()  # Display user instructions
        while True:
            ret, img = cam.read()  # Capture a frame from the webcam
            if not ret:
                print("Failed to grab frame")
                break

            image = cv.flip(img, 1)  # Flip the image horizontally
            filter_image = cv.addWeighted(image, 0.8, filter_frame, 0.3, 1)  # Blend the image with the filter
    
            cv.imshow("Filter", filter_image)  # Display the filtered image
    
            key = cv.waitKey(20)  # Wait for a key press

            if key == ord('x'):  
                break
            
            if key == ord('c'):
                save_image(filter_image, "MyPictures", "Filter_")
                print("Filtered selfie saved")
    finally:
        # Always release the camera and close all OpenCV windows
        cam.release()
        cv.destroyAllWindows()


def face_detection():
    # Function to detect faces in a webcam feed and draw rectangles around them

    user_guide()
    cam = cv.VideoCapture(0)
    try:
        while True:
            ret, image = cam.read()
            if not ret:
                print("Failed to grab frame!")
                break

            # Detect faces in the frame
            faces = classifier.detectMultiScale(image, 1.1, 5) 

            try:
                # Draw rectangles around detected faces
                for (x, y, w, h) in faces:
                    cv.rectangle(image, (x, y), (x + w, y + h), (200, 100, 50), 4)
            except:
                pass  # Ignore errors if face detection fails

            image = cv.flip(image, 1)
            cv.imshow("Face detection", image)
            key = cv.waitKey(20)

            if key == ord('x'):
                break
            
            if key == ord('c'):
                save_image(image, "MyPictures", "FaceDetection_")
                print("Face detected image saved")
    finally:
        cam.release()
        cv.destroyAllWindows()


def edge_detection():
    # Function to apply edge detection on a webcam feed and display the result       

    user_guide()
    cam = cv.VideoCapture(0)
    try:
        while True:
            ret, img = cam.read()
            if not ret:
                print("Failed to grab frame")
                break

            image = cv.flip(img, 1)
            
            # Apply Canny edge detection
            edge_image = cv.Canny(image, 100, 200)
    
            cv.imshow("Edge detection", edge_image)
            key = cv.waitKey(20)

            if key == ord('x'):
                break
            
            if key == ord('c'):
                save_image(edge_image, "MyPictures", "EdgeDetection_")
                print("Edge detected image saved")
    finally:
        cam.release()
        cv.destroyAllWindows()

        
def brightness_control(adjust):
    # Function to adjust the brightness of a webcam feed
    
    cam = cv.VideoCapture(0)
    try:
        while True:
            ret, img = cam.read()
            if not ret:
                print("Failed to grab frame")
                break

            image = cv.flip(img, 1)

            if adjust == "lower":
                # Decrease brightness
                adjusted_image = image.copy() * 0.7  # decrease by 30%
                adjusted_image[adjusted_image < 0] = 0  # Ensure pixel values don't go below 0
                adjusted_image = adjusted_image.astype(np.uint8)
            elif adjust == "higher":
                # Increase brightness
                adjusted_image = image.copy() * 1.5  # increase by 50%
                adjusted_image[adjusted_image > 255] = 255  # Ensure pixel values don't exceed 255
                adjusted_image = adjusted_image.astype(np.uint8)
            else:
                print("Invalid adjustment type!")
                return
            
            cv.imshow("Original", image)
            cv.imshow("Brightness adjusted", adjusted_image)
            
            key = cv.waitKey(20)

            if key == ord('x'):  # Terminate if input: x
                break
            
            if key == ord('c'):  # Take selfie and store it
                save_image(adjusted_image, "MyPictures", "BrightnessAdjusted_")
                print("Brightness adjusted image saved")
    finally:
        cam.release()
        cv.destroyAllWindows()



def face_blur():
    # Function to blur detected faces in a webcam feed

    user_guide()
    cam = cv.VideoCapture(0)
    try:
        while True:
            __, image = cam.read()

            # Detect faces in the frame
            faces = classifier.detectMultiScale(image, 1.1, 5)
            
            # Handle potential errors or face detection issues
            faceCrop = None
            try:
                # Identify the largest face in the frame
                for face in faces:
                    if face[-1] == max(faces[:, -1]):
                        faceCrop = face
                        break
                
                if faceCrop is not None:
                    x = faceCrop[0]
                    y = faceCrop[1]
                    w = faceCrop[2]
                    h = faceCrop[3]
                    faceCrop = image[y-10:y+h, x:x+w, :]

                    # Blurring cropped face
                    blur_image = cv.blur(faceCrop, (16, 16))

                    # Apply the blurred face back onto the original image
                    image[y-10:y+h, x:x+w, :] = blur_image

            except:
                pass  # Ignore errors if face detection fails

            image = cv.flip(image, 1)
            cv.imshow("Blur face", image)
            
            key = cv.waitKey(30)

            if key == ord('x'):  # Terminate if input: x
                break
            
            if key == ord('c'):  # Take selfie and store it
                save_image(image, "MyPictures", "BlurFace_")
                print("Blurred face image saved")
    finally:
        cam.release()
        cv.destroyAllWindows()
        

def masked_image():
    # Function to apply a color mask to the webcam feed and display the result

    user_guide()
    cam = cv.VideoCapture(0)
    lower = np.array([160, 180, 180]) # color for lower bound
    upper = np.array([255, 255, 255]) # color for upper bound
    try:
        while True:
            ret, img = cam.read()
            if not ret:
                print("Failed to grab frame")
                break

            image = cv.flip(img, 1)

            # Apply a color mask to the image to keep only pixels within the specified color range
            mask_img = cv.inRange(image, lower, upper)
            # Display the masked image where pixels within the color range are white, and others are black
            
            cv.imshow('Masked image', mask_img)
            
            key = cv.waitKey(20)

            if key == ord('x'):  
                break
            
            if key == ord('c'):  
                save_image(mask_img, "MyPictures", "MaskedImage_")
                print("Masked image saved")
    finally:
        cam.release()
        cv.destroyAllWindows()


def black_white():
    # Function to convert the webcam feed to black and white

    user_guide()
    cam = cv.VideoCapture(0)
    lower = np.array([180, 200, 200])
    upper = np.array([255, 255, 255])
    try:
        while True:
            ret, img = cam.read()
            if not ret:
                print("Failed to grab frame")
                break

            image = cv.flip(img, 1)

            # Convert to grayscale
            gray_image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
            cv.imshow('Black and white', gray_image)
            
            key = cv.waitKey(20)

            if key == ord('x'):  
                break
            
            if key == ord('c'):  
                save_image(gray_image, "MyPictures", "BlackAndWhite_")
                print("Black and white image saved")
    finally:
        cam.release()
        cv.destroyAllWindows()


def rgb_channels():
    # Open the camera

    print("To capture red frame, press: r")
    print("To capture green frame, press: g")
    print("To capture blue frame, press: b")
    print("To exit, press: x")
    print("." * 35)
    
    cam = cv.VideoCapture(0)

    try:
        while True:
            ret, img = cam.read()
            if not ret:
                print("Failed to grab frame")
                break

            img = cv.flip(img, 1)

            # Extract R, G, B channels by zeroing out the other channels
            r = img.copy()
            g = img.copy()
            b = img.copy()

            r[:, :, 1] = 0  # Zero out the green channel
            r[:, :, 2] = 0  # Zero out the blue channel

            g[:, :, 0] = 0  # Zero out the blue channel
            g[:, :, 2] = 0  # Zero out the red channel

            b[:, :, 0] = 0  # Zero out the green channel
            b[:, :, 1] = 0  # Zero out the red channel

            # reshaping image size to (400,300)
            img = cv.resize(img, (400, 300))  
            r = cv.resize(r, (400, 300))
            g = cv.resize(g, (400, 300))  
            b = cv.resize(b, (400, 300))
            
            # Display the color-separated frames
            cv.imshow("Red Channel", r)
            cv.imshow("Green Channel", g)
            cv.imshow("Blue Channel", b)

            key = cv.waitKey(20)
            
            if key == ord('x'): 
                break

            if key == ord('r'):  # for saving red frame, press: r
                save_image(r, "MyPictures", "RedFrame_")
                print("Red frame image saved")
            elif key == ord('g'):  # for saving green frame, press: g
                save_image(g, "MyPictures", "GreenFrame_")
                print("Green frame image saved")
            elif key == ord('b'):  # for saving blue frame, press: b
                save_image(b, "MyPictures", "BlueFrame_")
                print("Blue frame image saved")

    finally:
        cam.release()
        cv.destroyAllWindows()


print("#"*50)
print()
print(" "*12,"... Camera program ...")

run=1  # for while loop condition
choice=0  # for menu option choice

while(run == 1):
    
    # Display menu options
    print("#" * 50)
    print()
    print(">>> Options:\n")
    print("1: Take selfie")
    print("2: Filter")
    print("3: Face detection")
    print("4: Edge detection")
    print("5: Adjust brightness")
    print("6: Face blurring")
    print("7: Masked image")
    print("8: Monochrome (Black & White)")
    print("9: Color extraction from image")
    print("0: Exit\n")

    # Get user choice
    choice = input("Enter choice: ")
    print("*" * 50)
    
    if choice == '0':
        # Exit the loop and end the program
        
        run = 0
        print(" " * 15, "... Exit ...")
        print("#" * 50)
        break
    
    if choice == '1':
        # Option to take a selfie
        
        print("Your choice: selfie")
        print("-"*40)
        take_selfie()
        
    elif choice == '2':
        # Option to choose and apply a filter
        print("Your choice: selfie")
        print("-"*40)
        print()
        print(">>> Filter types:")
        print("1. cool(blue) filter")
        print("2. warm(yellow) filter")
        print("3. mixed color filter")
        print()
        
        filter_choice = input("Choose filter type: ")
        if filter_choice == '1':
            # Apply cool (blue) filter
            print("You selected: cool filter")
            print("." * 30)
            blue = [252, 215, 139]  # Light blue color
            filter_selfie(blue, "color")
        elif filter_choice == '2':
            # Apply warm (yellow) filter
            print("You selected: warm filter")
            print("." * 30)
            yellow = [139, 206, 247]  # Light yellow color
            filter_selfie(yellow, "color")
        elif filter_choice == '3':
            # Apply mixed color (theme) filter
            print("You selected: theme filter")
            print("." * 30)
            filter_selfie("mixed", "theme")
        else:
            print("!"*50)
            print()
            print("Invalid choice!")

    elif choice == '3':
        # Option for face detection
        face_detection()
        
    elif choice == '4':
        # Option for edge detection
        
        print("Your choice: edge detection")
        print("-"*40)
        edge_detection()
        
    elif choice == '5':
        # Option to adjust brightness

        print("Your choice: selfie")
        print("-"*40)
        print()
        print(">>> Brightness adjustment:")
        print("1. lower brightness")
        print("2. increase brightness")
        print()
        
        filter_choice = input("Choose filter type: ")
        
        if filter_choice == '1':
            # Lower brightness
            print("You selected: lower brightness")
            print("." * 30)
            brightness_control("lower")
        elif filter_choice == '2':
            # Increase brightness
            print("You selected: increase brightness")
            print("." * 30)
            brightness_control("higher")
        else:
            print("Invalid choice!")
            
    elif choice == '6':
        # Option to blur faces

        print("Your choice: face blurring")
        print("-"*40)
        face_blur()
        
    elif choice == '7':
        # Option to apply a color mask
        
        print("Your choice: image masking")
        print("-"*40)
        masked_image()
        
    elif choice == '8':
        # Option to convert to black and white

        print("Your choice: black and white")
        print("-"*40)
        black_white()
        
    elif choice == '9':
        # Option for color extraction from image

        print("Your choice: r,g,b color extraction")
        print("-"*40)
        rgb_channels()
        
    else:
        # Handle invalid choices
        print()
        print("!!! Invalid choice! Try Again...")
        print()

plt.show() # Show any remaining plots


