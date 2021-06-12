import tensorflow as tf
import numpy as np
import cv2

model_path = "posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite"
template_path = "person.jpg"
# target_path = "person_sit.jpeg"

# Load TFLite model and allocate tensors (memory usage method reducing latency)
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output tensors information from the model file
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

template_image_src = cv2.imread(template_path)
# src_tepml_width, src_templ_height, _ = template_image_src.shape 
template_image = cv2.resize(template_image_src, (width, height))
cv2.imshow("",template_image)

# add a new dimension to match model's input
template_input = np.expand_dims(template_image.copy(), axis=0)

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
print("template_heatmaps' shape:", template_heatmaps.shape)
print("template_offsets' shape:", template_offsets.shape)

# # Process target image. Same commands
# interpreter.set_tensor(input_details[0]['index'], target_input)
# interpreter.invoke()
# target_output_data = interpreter.get_tensor(output_details[0]['index'])
# target_offset_data = interpreter.get_tensor(output_details[1]['index'])
# target_heatmaps = np.squeeze(target_output_data)
# target_offsets = np.squeeze(target_offset_data)

# The output consist of 2 parts:
# - heatmaps (9,9,17) - corresponds to the probability of appearance of 
# each keypoint in the particular part of the image (9,9)(without applying sigmoid 
# function). Is used to locate the approximate position of the joint
# - offset vectors (9,9,34) is called offset vectors. Is used for more exact
#  calculation of the keypoint's position. First 17 of the third dimension correspond
# to the x coordinates and the second 17 of them correspond to the y coordinates

def parse_output(heatmap_data,offset_data, threshold):

  '''
  Input:
    heatmap_data - hetmaps for an image. Three dimension array
    offset_data - offset vectors for an image. Three dimension array
    threshold - probability threshold for the keypoints. Scalar value
  Output:
    array with coordinates of the keypoints and flags for those that have
    low probability
  '''

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
                cv2.circle(show_img,(int(round(kps[i,1]*ratio[1])),int(round(kps[i,0]*ratio[0]))),2,(0,255,255),round(int(1*ratio[1])))
                continue
        cv2.circle(show_img,(kps[i,1],kps[i,0]),2,(0,255,255),-1)
    return show_img

template_show = np.squeeze((template_input.copy()*127.5+127.5)/255.0)
template_show = np.array(template_show*255,np.uint8)
template_kps = parse_output(template_heatmaps,template_offsets,0.3)
cv2.imshow("draw",draw_kps(template_show.copy(),template_kps))
cv2.waitKey(0)
# # Matching keypoints indices in the output of PoseNet
# # 0. Left shoulder to right shoulder (5-6)
# # 1. Left shoulder to left elbow (5-7)
# # 2. Right shoulder to right elbow (6-8)
# # 3. Left elbow to left wrist (7-9)
# # 4. Right elbow to right wrist (8-10)
# # 5. Left hip to right hip (11-12)
# # 6. Left shoulder to left hip (5-11)
# # 7. Right shoulder to right hip (6-12)
# # 8. Left hip to left knee (11-13)
# # 9. Right hip to right knee (12-14)
# # 10. Left knee to left ankle (13-15)
# # 11.  Right knee to right ankle (14-16)

# parts_to_compare = [(5,6),(5,7),(6,8),(7,9),(8,10),(11,12),(5,11),(6,12),(11,13),(12,14),(13,15),(14,16)]

# def angle_length(p1, p2):

#   '''
#   Input:
#     p1 - coordinates of point 1. List
#     p2 - coordinates of point 2. List
#   Output:
#     Tuple containing the angle value between the line formed by two input points 
#     and the x-axis as the first element and the length of this line as the second
#     element
#   '''

#   angle = math.atan2(- int(p2[0]) + int(p1[0]), int(p2[1]) - int(p1[1])) * 180.0 / np.pi
#   length = math.hypot(int(p2[1]) - int(p1[1]), - int(p2[0]) + int(p1[0]))
  
#   return round(angle), round(length)

# template_values = []
# for part in parts_to_compare:
#   template_values.append(angle_length(template_kps[part[0]][:2], template_kps[part[1]][:2]))
# template_values

# # target_values = []
# # for part in parts_to_compare:
# #   target_values.append(angle_length(target_kps[part[0]][:2], target_kps[part[1]][:2]))
# # target_values

# def matching(template_kp, angle_deviation=30, size_deviation=1):

#   '''Input:
#       1. template_kp - list of tuples (for the template image) containng angles 
#       between particular body parts and x-axis as first elements and its sizes 
#       (distances between corresponding points as second elements)
#       2. target_kp - same for the target image
#       3. angle_deviation - acceptable angle difference between corresponding 
#       body parts in the images
#       4. size_deviation - acceptable proportions difference between the images
#     Output:
#       List of body parts which are deviated
#   '''

#   devs = []

#   # set an anchor size for proportions calculations - distance between shoulders
#   templ_anchor = template_kp[0][1]

#   # for each body part that we calculated angle and size for
#   for i in range(len(template_kp)):

#     angles = (template_kp[i][0], target_kp[i][0])
#     diff_angle = max(angles) - min(angles)

#     templ_size = (template_kp[i][1],templ_anchor)
#     templ_size = abs(min(templ_size) / max(templ_size))

#     tar_size = (target_kp[i][1], targ_anchor)
#     tar_size = abs(min(tar_size) / max(tar_size))

#     if diff_angle > angle_deviation:
#       devs.append(i)
#       print("{0} has different angle".format(i))

#     elif max(tar_size,templ_size) - min(tar_size,templ_size) > size_deviation:
#       devs.append(i)
#       print("{0} has different size".format(i))

#   return devs

# deviations = matching(template_values, target_values)
# deviations

# def draw_deviations(img, keypoints, pairs, deviations):

#   for i, pair in enumerate(pairs):

#     if i in deviations:
#       color = (0,0,255)
#     else:
#       color = (0,255,0)
      
#     cv2.line(img, (keypoints[pair[0]][1], keypoints[pair[0]][0]), (keypoints[pair[1]][1], keypoints[pair[1]][0]), color=color, lineType=cv2.LINE_AA, thickness=1)

# draw_deviations(target_show, target_kps, parts_to_compare, deviations)

# cv22_imshow(target_show)