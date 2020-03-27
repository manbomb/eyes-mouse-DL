import numpy as np
import argparse
import cv2
import pyautogui as pa

mx = np.zeros(3, dtype=int)
my = np.zeros(3, dtype=int)

mx_int = 0
my_int = 0

box_x = 0
box_y = 0

clock = 0

s_width = pa.size().width
s_height = pa.size().height

#print('Width: '+str(s_width/2)+' Height: '+str(s_height/2))

pa.moveTo(s_width/2,s_height/2)

def direct(image,i):
	nivel = i+1
	font                   = cv2.FONT_HERSHEY_SIMPLEX
	bottomLeftCornerOfText = (25,40*nivel)
	fontScale              = 1
	color             	   = (255,255,255)
	lineType               = 2

	if i == 0:
		#up
		cv2.putText(image, 'UP', bottomLeftCornerOfText, font, fontScale, color, lineType)
		pa.moveRel(0,-50)
	if i == 1:
		#down
		cv2.putText(image, 'DOWN', bottomLeftCornerOfText, font, fontScale, color, lineType)
		pa.moveRel(0,50)
	if i == 2:
		#left
		cv2.putText(image, 'RIGHT', bottomLeftCornerOfText, font, fontScale, color, lineType)
		pa.moveRel(50,0)
	if i == 3:
		#right
		cv2.putText(image, 'LEFT', bottomLeftCornerOfText, font, fontScale, color, lineType)
		pa.moveRel(-50,0)

	return image

ap = argparse.ArgumentParser()

ap.add_argument("-p", "--prototxt", required=True, help="arquivo .proto.txt")
ap.add_argument("-m", "--model", required=True, help="arquivo .caffemodel")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="probabilidade minima de corte")

args = vars(ap.parse_args())

captura = cv2.VideoCapture(0)

print("[INFO] carregando modelo...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

while(True):
	ret, frame = captura.read()

	image = frame.copy()

	(h, w) = frame.shape[:2]

	for ch in range(0,3):
		channel = frame[:,:,ch]
		channel = cv2.equalizeHist(channel)
		frame[:,:,ch] = channel

	frame = cv2.resize(frame, (300, 300))

	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

	net.setInput(blob)
	detections = net.forward()

	for i in range(0, detections.shape[2]):

		confidence = detections[0, 0, i, 2]

		if confidence > args["confidence"]:

			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			
			(startX, startY, endX, endY) = box.astype("int")

			w_face = endX-startX

			text = str(confidence*100)[0:4]+'%'

			y = startY - 10 if startY - 10 > 10 else startY + 10

			for j in range(1,3):
				mx[j] = mx[j-1]
				my[j] = my[j-1]

			mx[0] = (startX+endX)/2
			my[0] = (startY+endY)/2

			mx_int = int(mx.mean())
			my_int = int(my.mean())

			cv2.rectangle(image, (mx_int, my_int), (mx_int+2, my_int+2), (0, 0, 255), 2)

			if clock > 12 and box_y != 0 and box_x != 0:
				cv2.rectangle(image, (box_x-int(w_face*0.15), box_y-int(w_face*0.12)), (box_x+int(w_face*0.15), box_y+int(w_face*0.12)), (0, 0, 255), 2)

				if mx_int < box_x-int(w_face*0.15):
					print('RIGHT')
					direct(image,2)

				if mx_int > box_x+int(w_face*0.15):
					print('LEFT')
					direct(image,3)

				if my_int < box_y-int(w_face*0.15):
					print('TOP')
					direct(image,0)

				if my_int > box_y+int(w_face*0.15):
					print('BOTTOM')
					direct(image,1)

			cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 1)

			cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

	if clock%3 == 0:
		box_y = my_int
		box_x = mx_int

	cv2.imshow("WebCam", image)

	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break

	clock = clock+1