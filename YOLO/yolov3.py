import cv2
import numpy as np
import time

def get_coco_names():
    with open("data/coco.names","r") as f:
        names = f.read().split("\n")
    return names

def get_boxes(outputs,height_,width_):
    results = []
    boxes=[]
    confidences = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence>0.7:
                #print(detection[0],detection[1],detection[2],detection[3])
                width = int(detection[2]*width_)
                height = int(detection[3]*height_)
                detection[0] = int(detection[0]*width_)
                detection[1] = int(detection[1]*height_)
                x = int(detection[0]-(width/2))
                y = int(detection[1]-(height/2))
                results.append({
                    "geometry":[x,y,width,height],
                    "confidence": confidence,
                    "class_id": class_id
                })
                boxes.append([x,y,width,height])
                confidences.append(float(confidence))
    return results,boxes,confidences

def draw_boxes(img,boxes,indexes,classes):
    if len(tuple(indexes))==0:
        return img
    for i in indexes.flatten():
        box=boxes[i]
        x = box["geometry"][0]
        y = box["geometry"][1]
        w = box["geometry"][2]
        h = box["geometry"][3]
        rnd_cl = tuple([np.random.randint(0,255) for i in range(3)])
        img=cv2.rectangle(img,(x,y),(x+w,y+h),color=rnd_cl,thickness=2)
        img=cv2.putText(img,classes[box["class_id"]]+":"+str(box["confidence"]),(x+w,y-1),1,1,color=(255,0,0))
    return img

def detect_objects(test_img):
    st_tm = time.time()
    img_ht,img_wd = test_img.shape[:2]
    blob = cv2.dnn.blobFromImage(test_img,1/255,(416,416),(0,0,0),swapRB=True,crop=False)
    model.setInput(blob)
    layer = model.getUnconnectedOutLayersNames()
    layeroutput = model.forward(layer)

    #Draw
    result,boxes,confidences = get_boxes(layeroutput,img_ht,img_wd)
    indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)
    end_tm = time.time()
    test_result = draw_boxes(test_img,result,indexes,classes)
    return test_result,(end_tm - st_tm)

#Video Demo
def webcam_demo():
    cap = cv2.VideoCapture("YOLO/market.mp4")

    while True:
        _,frame = cap.read()
        if not _:
            break
        #frame = cv2.flip(frame,1)
        res_frame,t = detect_objects(frame)
        print(res_frame.shape)
        fps = np.round((1/t),3)
        res_frame = cv2.putText(res_frame,"FPS:"+str(fps),(res_frame.shape[1]-80,10),1,1,(255,0,0))
        cv2.imshow("Object Detection",res_frame)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break


model = cv2.dnn.readNet("D:/Desktop/Projects/Mega/data/yolov3.weights","D:/Desktop/Projects/Mega/data/yolov3.cfg")
model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
count = cv2.cuda.getCudaEnabledDeviceCount()
print(count)
#Get coco classes
classes = get_coco_names()
webcam_demo()
# test_img = cv2.imread('YOLO/bridge.jpg')
# result_img = test_img.copy()
# result_img = detect_objects(result_img)
# cv2.imshow("Original",test_img)
# cv2.imshow("Result",result_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()