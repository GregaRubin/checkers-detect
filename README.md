# Checkers Detect

## A Python command line script for detecting the position and type of checkers pieces in an image.
Checkers detect uses a neural network model made with Tensorflow Keras to detect which type of the 4 checkers pieces we are dealing with (white/black man and white/black king).
The model was trained on a dataset of around 1000 images of different board pieces. Our images were labeled using Label Studio (https://labelstud.io/) and some of them were augmented before training our nerural network model.\


![centers](https://github.com/user-attachments/assets/53de1d09-6c30-4014-9bbc-1978538af057)

The script starts by opening up a pyplot and asks you to select the 4 corners of your checkers board.
Once the corners are select it calculates each possible position of a board piece using perspectiveTransform from openCV. Each board piece image is evaulated using the trained neural network and saved in an array if it qulifies as a board piece.

![predict](https://github.com/user-attachments/assets/af8e8e32-d475-40f1-8cf4-879da17bc753)


A board.json file is created which contains a 8x8 json array with board piece names ("man_white", "man_black", "king_white", "king_black") or empty spots ("X") on each index. The first element of the array represents the board position A8 
and the last element represent H1.

## Installation
1. Download and install Python and pip (https://www.python.org/downloads/)
2. Clone the GitHub repository
3. Run 'pip install -r requirements.txt' inside the project folder
5. Run the script

## Running the script
+You can run the script with the command 'python checkersDetect.py [path_to_image]'
+The board.json file is created inside the current working directory
