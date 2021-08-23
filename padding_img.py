import os
import cv2
files=os.listdir("./image/")
for name in files:
    
    image = cv2.imread("./image/"+name)
    h,w=image.shape[0],image.shape[1]
    scale=h/32
    resized_w=int(w/scale)
    image = cv2.resize(image, (resized_w, 32))
    if 128>w:
        padding=(128-w)//2
        image=cv2.copyMakeBorder(image,padding,0,0,padding,128-w-padding,cv2.BORDER_CONSTANT,value=(0,0,0))
        
    if 128<w:
        
        image=cv2.resize(image, (128, 32))
    cv2.imwrite("./image/"+name,image)
    #assert False
    
