import cv2
import numpy as np

#Carregar o YOLO

redeNeural = cv2.dnn.readNet("yolov3.weights", 'yolov3.cfg') #precisa dos pesos da rede Neural, yolov3.weights

classes = []
with open("coco.names", 'r') as file:
    classes = [line.strip() for line in file.readlines()]  #colocando o coconames em um array

layer_names = redeNeural.getLayerNames()
previsores = [layer_names[i[0] - 1] for i in redeNeural.getUnconnectedOutLayers()]
colors = np.random.uniform(255,0, size=(len(classes), 3))

cap = cv2.VideoCapture(0)
while True:
    _,frame = cap.read()

    height, width, channels = frame.shape

    borrar = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    # for b in borrar:
    #     for n, img_borrada in enumerate(b):
    #         cv2.imshow(str(n), img_borrada)      REPRESENTAÇÃO DO BLOB

    redeNeural.setInput(borrar)
    outs = redeNeural.forward(previsores)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4) #duplicado
    number_objects_detected = len(boxes)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes: #tirou o duplicado
            x, y, w, h = boxes[i]
            label = classes[class_ids[i]]
            color = colors[i]
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
            cv2.putText(frame, label, (x,y+30), font, 2, color, 2)

    cv2.imshow("test_rectangle", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
