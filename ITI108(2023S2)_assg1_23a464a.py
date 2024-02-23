import cv2
import os
import tensorflow as tf
#from tflite_runtime.interpreter import Interpreter
import tensorflow_hub as hub
import numpy as np
from controller import Robot
from controller import Keyboard
import time
import pandas as pd
import asyncio
import shutil
from keras import layers

# Define label list
threshold = 0.70
labels = ["left","right"]
output_folder1 = "gestures/left"
output_folder2 = "gestures/right"

 # create the Robot instance.
robot = Robot()
left_wheel = robot.getDevice('left wheel')
right_wheel = robot.getDevice('right wheel')
left_wheel.setPosition(float('inf'))
right_wheel.setPosition(float('inf'))
left_wheel.setVelocity(0.0)
right_wheel.setVelocity(0.0)

CRUISING_SPEED= 9.0
TURN_SPEED = CRUISING_SPEED/2.0
TIME_STEP = 100

def inference():
# # Start video capture
    cap2 = cv2.VideoCapture(0)

    while True:
        ret, frame = cap2.read()

        if not ret:
            break
        # Resize capturing frame
        frame = cv2.resize(frame, (224, 224))
        interpreter = tf.lite.Interpreter(model_path="hands_quant.tflite")

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.resize_tensor_input(input_details[0]['index'], (1,224, 224,3))
        interpreter.resize_tensor_input(output_details[0]['index'], (1,2))
        interpreter.allocate_tensors()

        # Preprocess the frame
        #frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #frame_resized = cv2.resize(frame, (224, 224))  # Assuming model input size
        input_data = np.expand_dims(frame, axis=0).astype(np.float32)
        input_data = input_data / 255.0
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]["index"], input_data)
        
        # Run inference
        interpreter.invoke()
        # Get output probabilities
        output_data = interpreter.get_tensor(output_details[0]["index"])

        tflite_pred_dataframe = pd.DataFrame(output_data)
        tflite_pred_dataframe.columns = labels
        print(tflite_pred_dataframe)

        highest_probability = tflite_pred_dataframe.max().max()

        if highest_probability >= threshold:
            predicted_label_index = np.argmax(output_data[0])
            predicted_label = labels[predicted_label_index]  # Assuming labels are defined globally
            print("predicted_label: " , predicted_label)
        else:
            predicted_label = 'straight'
            print("No prediction made (threshold not met)")

        cv2.putText(frame, f"P:{predicted_label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Real time inference", frame)

        if cv2.waitKey(1) == ord("q"):
                break
        
        #state= {"predicted_label":predicted_label}
        #print("state =",state)
        TIME_STEP=10
        while robot.step(TIME_STEP) != -1:
            if(predicted_label == "left"):#if(key == ord('W')):
                left_wheel.setVelocity(-TURN_SPEED)
                right_wheel.setVelocity(TURN_SPEED)
                break
            elif (predicted_label == "right"):#elif (key ==  ord('D')):
                left_wheel.setVelocity(TURN_SPEED)
                right_wheel.setVelocity(-TURN_SPEED)
                break
            else:
                left_wheel.setVelocity(CRUISING_SPEED)
                right_wheel.setVelocity(CRUISING_SPEED)
                break
        time.sleep(0.1)

    cap2.release()
    cv2.destroyAllWindows()

    #return predicted_label

# Function to capture images for a training of model
def capture_images(gesture_name, num_images):
    count = 0
    cap = cv2.VideoCapture(0)
    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.putText(frame, f"{count + 1}/{num_images}", (10, 30), #{gesture_name}
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Resize to 224x224 
        frame = cv2.resize(frame, (224, 224))

        #display frame
        cv2.imshow(f"For {gesture_name}. 'c' to capture, 'x'to delete last ", frame)
        key = cv2.waitKey(1) & 0xFF

        #wait for key press
        if key == ord("c"):
            cv2.imwrite(f"gestures/{gesture_name}/{count}.jpg", frame)
            count += 1
        elif key == ord("x"):  # Delete last captured image
            if count > 0:
                os.remove(f"gestures/{gesture_name}/{count - 1}.jpg")
                count -= 1
        elif key == ord("e"):  # Complete capture
            break
    cap.release()
    cv2.destroyAllWindows()

# Function to train model tensorflow model .pd and quant tflitemodel.
def train_model():

    #delete past records and create new folders
    SAVED_MODEL = "saved_model/"
    shutil.rmtree(SAVED_MODEL)
    os.remove("hands_quant.tflite")
    os.makedirs(SAVED_MODEL, exist_ok=True)

    #path to folder for left/right foldering of images
    data_root = 'gestures/'

    #setting input dimension and path
    IMAGE_SHAPE = (224, 224)
    TRAINING_DATA_DIR = str(data_root)

    #scalding the pixel to between 0 to 1. 
    datagen_kwargs = dict(rescale=1./255, validation_split=.20)

    #train / val split
    valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)
    valid_generator = valid_datagen.flow_from_directory(
    TRAINING_DATA_DIR,
    subset="validation",
    shuffle=True,
    target_size=IMAGE_SHAPE)
    
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)
    train_generator = train_datagen.flow_from_directory(
    TRAINING_DATA_DIR,
    subset="training",
    shuffle=True,
    target_size=IMAGE_SHAPE)    

    # labelling checks
    print(train_generator.class_indices.items())
    dataset_labels = sorted(train_generator.class_indices.items(), key=lambda pair:pair[1])
    print(dataset_labels)
    dataset_labels = np.array([key.title() for key, value in dataset_labels])
    print(dataset_labels)

    #downloading mobile net feature extractor model without the classifier head
    model =tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4",
                    output_shape=[1280],
                    trainable=False),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(train_generator.num_classes, activation='softmax')
    ])
    model.build([None, 224, 224, 3])

    model.compile(
    optimizer=tf.keras.optimizers.legacy.Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy'])

    steps_per_epoch = np.ceil(train_generator.samples/train_generator.batch_size)
    val_steps_per_epoch = np.ceil(valid_generator.samples/valid_generator.batch_size)

    hist = model.fit(
    train_generator,
    epochs= 5,
    verbose=1,
    steps_per_epoch=steps_per_epoch,
    validation_data=valid_generator,
    validation_steps=val_steps_per_epoch).history

    # Print the validation accuracy
    final_loss, final_accuracy = model.evaluate(valid_generator, steps = val_steps_per_epoch)
    print("Final loss: {:.2f}".format(final_loss))
    print("Final accuracy: {:.2f}%".format(final_accuracy * 100))

    #save model
    tf.saved_model.save(model, SAVED_MODEL)

    # Convert the model to TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quant_model = converter.convert()
    open("hands_quant.tflite", "wb").write(tflite_quant_model)
    print("hands_quant.tflite model updated")

 

# Main function
def main():
    while True:
        choice = input("Enter 1 to retrain model, 2 to steer robot: ")
        if choice == "1":
            
            #delete all images
            shutil.rmtree(output_folder1)
            shutil.rmtree(output_folder2)
            os.makedirs(output_folder1, exist_ok=True)
            os.makedirs(output_folder2, exist_ok=True)

            # Capture images for left and right folder
            for gesture_name in ["left", "right"]: #,"straight"
                capture_images(gesture_name, 60)  # Adjust number of images

            # Train the model
            train_model()

        elif choice == "2":
            
            inference()

            
           

if __name__ == "__main__":
    asyncio.run(main())