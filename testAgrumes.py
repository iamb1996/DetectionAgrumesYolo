# import required packages
import cv2
import argparse
import numpy as np
import os

# handle command line arguments
# read input image

conf = 0.6
list=os.listdir("cropped_images")
for kk in list:
    comptage = 0
    image = cv2.imread('cropped_images/'+str(kk)) #chemin dossier a tester
    if image is None:
        while image is None:
            kk+=1
            image = cv2.imread('cropped_images/'+str(kk))
    print('cropped_images/'+str(kk))
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392
	    # read class names from text file
    classes = None
    with open('tools/obj.names', 'r') as f:
	    classes = [line.strip() for line in f.readlines()]

	    # generate different colors for different classes 
	    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

	    # read pre-trained model and config file
	    net = cv2.dnn.readNet(r'C:\Users\hp\Documents\Traitement\FINAL_WEIGHTS\tiny-yolo_new_anchors_final.weights', r'C:\Users\hp\Documents\Traitement\tools\tiny-yolo_new_anchors.cfg')
	    #net = yolo.YoloV4(
            #names_path=r"C:\Users\hp\Documents\Traitement\tools\obj.names",
            #weights_path=r"C:\Users\hp\Documents\YOLO4\weights\yolov4-custom_final.weights")

	    # create input blob 
	    blob = cv2.dnn.blobFromImage(image, scale, (Width,Height), (0,0,0), True, crop=False)

	    # set input blob for the network
	    net.setInput(blob)

	    def get_output_layers(net):
	        
	        layer_names = net.getLayerNames()
	        
	        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

	        return output_layers

	    # function to draw bounding box on the detected object with class name
	    def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

	        label = str(classes[class_id])

	        color = COLORS[class_id]
	       

	        cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), (0,0,255), 5)

	        cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)


	    outs = net.forward(get_output_layers(net))

	    # initialization
	    class_ids = []
	    confidences = []
	    boxes = []
	    conf_threshold = 0.01
	    nms_threshold = 0.01

	    # for each detetion from each output layer 
	    # get the confidence, class id, bounding box params
	    # and ignore weak detections (confidence < 0.5)
	    for out in outs:
	        for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > conf:
                        center_x = int(detection[0] * Width)
                        center_y = int(detection[1] * Height)
                        w = int(detection[2] * Width)
                        h = int(detection[3] * Height)
                        x = center_x - w / 2
                        y = center_y - h / 2
                        class_ids.append(class_id)
                        confidences.append(float(confidence))
                        boxes.append([x, y, w, h])
                        print(center_x/256 ,center_y/256 ,w/256,h/256)
                        out = open("output/"+str(kk)+".txt", "a")
                        out.write("0 "+str(center_x/256)+" "+str(center_y/256)+" "+str(w/256)+" "+str(h/256)+"\n")
                        out.close()

	    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
	    # go through the detections remaining
	    # after nms and draw bounding box
	    for i in indices:
	        i = i[0]
	        box = boxes[i]
	        x = box[0]
	        y = box[1]
	        w = box[2]
	        h = box[3]
	        draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
	        comptage += 1
	        #cv2.imwrite("test_results/"+str(kk)+".jpg", image)

	    # display output image    
	    #cv2.imshow("object detection", image)

	    # wait until any key is pressed
	    #cv2.waitKey(0)
	        
	     # save output image to disk
    cv2.imwrite("test_results/"+str(kk)+".jpg", image)
    print(comptage)
    files = open("accuracy.txt", "a")
    files.write("For ing: "+str(kk)+" algorithm detected "+str(comptage)+"\n")
    files.close()
	    #cv2.destroyAllWindows()

	    

# jpg=os.listdir("test_results")
# txt=os.listdir("output")
# cpt=1
# for k in jpg:
# 
#     os.path.splitext(k)
#     d=os.path.splitext(k)[0]
#     d=str(d)+".txt"
#     print(k+"              "+d+"   "+str(cpt))
#     cpt+=1
#     if d not in txt:
#         print(d+"     "+k)
#         os.remove("test_results/"+k)
