import tensorflow as tf
import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d','--debug', help='Check debug mode', required=False)
args = vars(parser.parse_args())

def parse_output(heatmap_data,offset_data, threshold):

  joint_num = heatmap_data.shape[-1]
  pose_kps = np.zeros((joint_num,3), np.uint32)

  for i in range(heatmap_data.shape[-1]):

      joint_heatmap = heatmap_data[...,i]
      max_val_pos = np.squeeze(np.argwhere(joint_heatmap==np.max(joint_heatmap)))
      remap_pos = np.array(max_val_pos/8*257,dtype=np.int32)
      pose_kps[i,0] = int(remap_pos[0] + offset_data[max_val_pos[0],max_val_pos[1],i])
      pose_kps[i,1] = int(remap_pos[1] + offset_data[max_val_pos[0],max_val_pos[1],i+joint_num])
      max_prob = np.max(joint_heatmap)

      if max_prob > threshold:
        if pose_kps[i,0] < 257 and pose_kps[i,1] < 257:
          pose_kps[i,2] = 1

  return pose_kps

def draw_kps(show_img,kps, ratio=None):
    for i in range(5,kps.shape[0]):
        if kps[i,2]:
            if isinstance(ratio, tuple):
                cv2.circle(show_img,(int(round(kps[i,1]*ratio[1])),int(round(kps[i,0]*ratio[0]))),2,(0,0,0),round(int(1*ratio[1])))
                continue
        cv2.rectangle(show_img,(kps[i,1]-10,kps[i,0]+10),(kps[i,1]+10,kps[i,0]-10),(0,0,0),-1)
    return show_img
  
model_path = "posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite"
cap = cv2.VideoCapture(0)

# Load TFLite model and allocate tensors (memory usage method reducing latency)
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output tensors information from the model file
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

while True:
  try:
    _,img_src = cap.read()
    # src_tepml_width, src_templ_height, _ = img_src.shape 
    img = cv2.resize(img_src, (width, height))
    imgClone = np.zeros(img.shape,dtype=np.uint8)
    imgClone.fill(255) # or img[:] = 255
    cv2.imshow("",img)

    # add a new dimension to match model's input
    template_input = np.expand_dims(img.copy(), axis=0)

    # check the type of the input tensor
    floating_model = input_details[0]['dtype'] == np.float32

    # Brings input values to range from 0 to 1
    if floating_model:
      template_input = (np.float32(template_input) - 127.5) / 127.5

    # Process template image
    # Sets the value of the input tensor
    interpreter.set_tensor(input_details[0]['index'], template_input)
    # Runs the computation
    interpreter.invoke()
    # Extract output data from the interpreter
    template_output_data = interpreter.get_tensor(output_details[0]['index'])
    template_offset_data = interpreter.get_tensor(output_details[1]['index'])
    # Getting rid of the extra dimension
    template_heatmaps = np.squeeze(template_output_data)
    template_offsets = np.squeeze(template_offset_data)

    template_show = np.squeeze((template_input.copy()*127.5+127.5)/255.0)
    template_show = np.array(template_show*255,np.uint8)
    template_kps = parse_output(template_heatmaps,template_offsets,0.3)
    if args["debug"]:
       cv2.imshow("Image",draw_kps(template_show.copy(),template_kps))
    else:
      cv2.imshow("Pose",draw_kps(imgClone,template_kps))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  except:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
