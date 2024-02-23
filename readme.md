# Transfer Learning Meets Real-Time User Control: A Case Study in Custom Model Retraining

Emphasizing Real-Time Responsiveness and Customization

This application harnesses the power of transfer learning and real-time adaptation to create intelligent, adaptable robot control. It enables users to seamlessly retrain a MobileNet V2 SSD model using images captured on the spot, ensuring the robot responds effectively to its specific environment. The tailored model then powers real-time video classification, guiding the robot to make precise left, right, or straight
.

This bot has been created using [Webot](https://cyberbotics.com/), the challenge is to navigate the bot into and out of the maze, using deep learning model to inference steering of the bot.

## folder structure
-23A464A.zip
    - controller (all the required webot libaries)
    - gestures (user captured images)
        -left
        -right
    - hands_quant.tflite (generate tf.lite model)
    - ITI108(2023S2)_assg1_23a464a.py (application file to run)
    - Model_report.docx (summary of model comparison)
    - readme.md (how to utilize the project)
    - requirements.txt (to install require python library)
    - saved_model (the tensorflow model)
    - 23a464a assignment 1 video.mov (Demo video of app usage)

## Prerequisites
This prerequisites in order to run local in Mac M1 os.

### this application works with Python 3.9

## Running the sample
- install tensorflow for Mac M1 OS to prevent "zsh: illegal hardware instruction". [youtube instruction guide](https://www.youtube.com/watch?v=WFIZn6titnc)  
- activated the tensorflow virtual enviroment
- Run `pip install -r requirements.txt` to install all dependencies
- install [Webot](https://cyberbotics.com/)
- setup webot config in Mac M1 os
    - Terminal
    1. nano ~/.zshrc
    2. create these variable at top of .zshrc
        - export WEBOTS_HOME="/Applications/Webots.app/Contents" # Assuming Webots is installed in /Applications
        - export PYTHONPATH="$WEBOTS_HOME/lib/controller/python"
        - PATH="$PATH:$WEBOTS_HOME/lib/controller"
        - PATH="$PATH:$WEBOTS_HOME/msys64/mingw64/bin"
        - PATH="$PATH:$WEBOTS_HOME/msys64/mingw64/bin/cpp"
        - export PYTHONIOENCODING="UTF-8"
    3. Save the changes:
        - Press Control+O to save the file.
        - Press Enter to confirm the filename.
        - Press Control+X to exit nano.
    4. Reload the profile:
        - source ~/.zshrc
        - to test variable
            - echo $WEBOTS_HOME
            - echo $PYTHONPATH
            - echo $PATH
            - echo $PYTHONIOENCODING
- copy "controller" folder from /Applications/Webots.app/Contents/lib/controller/python/controller to the same folder as `10ITI108(2023S2)_assg1_23a464a.py` 
- ensure Terminal has access to Camera function. Setting -> Privacy & Security -> Camera -> enable option for Terminal 
- run 'linefollower.wbt'
- Run `10ITI108(2023S2)_assg1_23a464a.py`


## Using the App(10ITI108(2023S2)_assg1_23a464a.py)

when 10ITI108(2023S2)_assg1_23a464a.py is launched.

### option 1 - Re-train the model
- a live stream window will pop up
- place your image or pose infront of the stream window
- press 'c' to capture the image, 'x' to delete last photo
- 60 images for left, followed by 60 images for the right. (stream window title will indicate left or right)
- model will start re-training automaticalling once all images are capture
- review the val-loss and val-acc score
- a quantized .tflite model will be generated

### option 2 - Steer Robot

- the robot will began to move straight
- place your left or right image onto the streaming window
- streaming window will indicate left or right direction is inference.
- inference is based on 70% confidence score. every inference score is available in the Terminal
- Turn your webot though the maze by flashing the required image
- Terminal Ctrl+ C to end

## Model compairson report
Model training compairson result is available at Model_report.docx. The report is to understand the validation score vs model file size across tensorflow .pd model vs .tflite vs quantized . tflite model

## Further reading
- [project inspiration](https://storage.googleapis.com/tfjs-examples/webcam-transfer-learning/dist/index.html)
- [Mobile net model download](https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4)
- [Webot download](https://cyberbotics.com/)
- [How to - zsh: illegal hardware instruction (TensorFlow m1 Mac)](https://www.youtube.com/watch?v=WFIZn6titnc)
