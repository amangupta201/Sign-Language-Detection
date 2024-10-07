# Sign-Language-Detection
This Sign Language Detection project uses computer vision to detect and classify American Sign Language (ASL) gestures in real-time. It supports 11 common ASL gestures, including symbols for "Hello," "Thank You," "I Love You," and more.

Labels:
Bathroom
Connection
Drink
Eat
Goodbye
Hello
I Love You
Thank You
Water
Yes
No
The project uses hand tracking to capture the hand region and feeds the cropped hand image to a pre-trained model for gesture recognition.

Installation
Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/sign-language-detection.git
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Download the pre-trained model: Place the keras_model.h5 and labels.txt in the Model/ directory.

Usage
Data Collection
To collect images for a specific sign (e.g., "Drink"), run:

bash
Copy code
python dataCollection.py
This will start a webcam stream. Make the desired sign in front of the camera and press the 's' key to capture an image. The captured images will be stored in the Data/ folder for later use in training.

Model Training
Train the model using Google's Teachable Machine by uploading the collected images and exporting the model as keras_model.h5.

Testing
To test real-time gesture recognition:

bash
Copy code
python Test.py
This will start a webcam stream. The system will detect hand gestures and classify them using the trained model. The recognized label will be displayed on the screen.

Example
Make the "Hello" gesture, and the system will recognize it and display the word "Hello" on the screen.
Requirements
Python 3.7+
OpenCV
cvzone
numpy
Install the dependencies with:

bash
Copy code
pip install -r requirements.txt
Model
The model was trained using images collected manually through the dataCollection.py script and Google's Teachable Machine.

Contributing
Feel free to fork this repository and make changes to extend the system to detect more gestures or improve accuracy.

License
This project is licensed under the MIT License.

By following this documentation, you'll be able to set up the project, collect data, and recognize ASL gestures using your webcam.
