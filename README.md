# Checkers Detect

## A Python command-line script for detecting the position and type of checkers pieces in an image.
Checkers Detect uses a neural network model built with TensorFlow Keras to detect which of the four types of checkers pieces we are dealing with (white/black man and white/black king). The model was trained on a dataset of around 1,000 images of different board pieces. Our images were labeled using Label Studio (https://labelstud.io/), and some of them were augmented before training our neural network model.


![centers](https://github.com/user-attachments/assets/53de1d09-6c30-4014-9bbc-1978538af057)

The script starts by opening up a pyplot window and asks you to select the four corners of your checkers board. Once the corners are selected, it calculates the centers of each possible board piece using perspectiveTransform from OpenCV. Each board piece image is cropped based on the board square width, evaluated using the trained neural network, and saved in an array if it qualifies as a board piece.

![predict](https://github.com/user-attachments/assets/af8e8e32-d475-40f1-8cf4-879da17bc753)


A board.json file is created, which contains an 8x8 JSON array with board piece names ("man_white", "man_black", "king_white", "king_black") or empty spots ("X") in each index. The first element of the array represents the board position A8, and the last element represents H1.

## Installation
1. Download and install Python and pip (https://www.python.org/downloads/)
2. Clone the GitHub repository
3. Run 'pip install -r requirements.txt' inside the project folder
5. Run the script

## Running the script
+ You can run the script with the command 'python checkersDetect.py [path_to_image]'
+ The board.json file is created inside the current working directory
