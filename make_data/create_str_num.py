#!/usr/bin/python
#coding:utf-8

import glob
import os
import random
import time
from uuid import uuid1
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter,ImageEnhance
import utils
import cv2
import threading
import string

'''中文图片生产器'''

'''宏部署'''
backPaths = glob.glob('./bg/*')  # 背景图像路径
fonts = glob.glob('./fonts/*')  # 字体集路径
font_num=len(fonts)
index=0
imgsnum = 0  # 图片计数个数
shownum = 1000  # 生成n张图片打印一次
imgsize = [860, 1024, 1600]  # 大图尺寸（无需修改，如要修改，请往大的改）
width = 128  # 生成图片实际宽度
height = 32  # 生成图片实际高度
root = './imgs_test' # 图片文本数据存放路径
batch = 10000# 生成图片批数（不是实际生成图片数量）

all_data = []

def randNum(low, high):
    return random.randint(low, high)

def make_bg():
    h=np.random.randint(300,500)
    w=np.random.randint(400,500)
    img = np.ones((h, w,3), dtype=np.uint8) 
    temp=random.randint(0,90)
    img[:,:,0]=img[:,:,0]*random.randint(240,255)
    img[:,:,1]=img[:,:,1]*temp
    img[:,:,2]=img[:,:,2]*temp
    #复杂背景
    '''
    #划线
    for _ in range(np.random.randint(500,800)):
        pt1 = (np.random.randint(0, w), np.random.randint(0, h))
        pt2 = (np.random.randint(0, w), np.random.randint(0, h))
        temp=np.random.randint(0, 90)
        #只要蓝色
        color = (temp, temp,temp)
        cv2.line(img, pt1, pt2, color, 1)
    #画矩形框
    for _ in range(np.random.randint(0,5)):
        pt1 = (np.random.randint(0, w), np.random.randint(0, h))
        pt2 = (np.random.randint(0, w), np.random.randint(0, h))
        temp = np.random.randint(0, 90)
        color=(temp,temp,temp)
        cv2.rectangle(img, pt1, pt2, color, -1)
    '''

    img = Image.fromarray(img)
    image_bright = ImageEnhance.Brightness(img)
    # 亮度减弱
    img =  image_bright.enhance(random.uniform(0.7,1.3))
    
    methods=[Image.NEAREST,Image.BILINEAR,Image.BICUBIC,Image.ANTIALIAS]
    img=img.resize((int(img.size[0]*random.uniform(1.0,3.0)),int(img.size[1]*random.uniform(1.0,3.0))),np.random.choice(methods))

    
    return img
  
def buider_bimg():
    # 图像背景生成
    temp = randNum(0, 10)
    
    flag = 0
    if temp >=3:
        p = make_bg()
        Size=p.size[0],p.size[1] #w,h
        #k=np.random.randint(0,10000)
        #p.save("./{}.jpg".format(k))
    else:
        bg = Image.open(np.random.choice(backPaths))
        image_bright = ImageEnhance.Brightness(bg)
        # 亮度减弱
        img =  image_bright.enhance(random.uniform(0.7,1.3))
        methods=[Image.NEAREST,Image.BILINEAR,Image.BICUBIC,Image.ANTIALIAS]
        img=img.resize((int(img.size[0]*random.uniform(1.0,1.5)),int(img.size[1]*random.uniform(1.0,1.5))),np.random.choice(methods))  
        p=img
        Size=p.size[0],p.size[1]
        flag = 1
    return p, Size, flag

def fromat_box(box):
    # 格式化坐标点
    a, b, a1, b1 = box
    if a1 - a < width and b1 - b < height:
        #resize后会变形
        spanx = (width - a1 + a) // 2
        bias_x=random.randint(0,spanx)
        x1, x2 = a - spanx-bias_x, a1 + spanx-bias_x
        spany = (height - b1 + b) // 2
        bias_y=random.randint(0,spany)
        y1, y2 = b - spany-bias_y, b1 + spany-bias_y
    elif a1 - a < width and b1 - b > height:
        #使长宽比符合280:32
        spanx = (width-a1+a)//2
        
        bias_x=random.randint(0,spanx)
        x1, x2 = a - spanx-bias_x, a1 + spanx-bias_x
        y1, y2 =b,b1
        
    elif a1 - a > width and b1 - b < height:
        x1, x2 = a, a1
        spany = int((a1 - a - width) / 2*(32/width))
        bias_y=random.randint(0,spany)
        y1, y2 = b - spany-bias_y, b1 + spany-bias_y
        
        
    else:
        spanx=(a1-a-width)//2
        spany=(b1-b-height)//2
        if spany*4>spanx:
            x1,x2=a-int(spany*4-spanx)//4,a+int(spany*4-spanx)//4
            y1,y2=b,b1
        if spany*4<spanx:
            x1,x2=a,a1
            y1,y2=b-int(spanx-spany*4)//4,b1+int(spanx-spany*4)//4
        else:
            x1,x2=a,a1
            y1,y2=b,b1
    return x1, y1, x2, y2
#语料生成，中文可以自己搜集中文语料，添加函数生成
def random_num(lengths=15):
    s=""
    time=random.randint(4,6)
    point=random.randint(1,time-1)
    for i in range(time):
        s=s+str(random.randint(0,9))
        
    s=s[:point]+"."+s[point:]        
    if random.randint(0,1):
        s="-"+s
    return s
def random_char(lengths=15):
    s=""
    time=random.randint(6,lengths)
    for i in range(time):
        s=s+random.choice(string.ascii_letters[26:])
        s=s+random.choice('ZENMWKXFPUJTYILDBEOQ')
        if len(s)>lengths:
            break
    return s
def random_char_num(lengths=15):
    s=""
    time=random.randint(6,lengths)
    for i in range(time):
        if random.randint(0,1):
            s=s+random.choice(string.ascii_letters[26:])
        else:
            s=s+random.choice(string.digits)
    return s
def draw_txt():
    # 绘制文字
    #txtlist = select_txt()
    txtlist =[]
    
    for i in range(5):
        txtlist.append(random_num())

    bimg, size, flag = buider_bimg()
    # print size
    X, Y = size#w,h
    initX, initY = int(size[0] * 0.1), int(size[1] * 0.1)

    textboxs = []
    imgBoxes = []

    draw = ImageDraw.Draw(bimg)
    global index
    fontType = fonts[index%font_num]  # 随机获取一种字体
    index+=1
    cX = initX
    cY = initY
    #print(txtlist)
    for lable in txtlist:
        fontSize = random.randint(25,30)  # 字体大小
        # fontSize=10
        font = ImageFont.truetype(fontType, fontSize)

        charW, charH = draw.textsize(text=lable, font=font)
        if cY+charH < Y - initY and cX+charW <= X - initX:
            color=random.randint(200,255)
            draw.text(xy=(cX, cY), text=lable, font=font, fill=(color,color,color))
            box = cX, cY, cX + charW, cY + charH
            x1, y1, x2, y2 = fromat_box(box)
            imgBoxes.append([x1, y1, x2, y2])
            
            textboxs.append(lable)
            cY += charH + random.choice(range(40,50))
            if random.choice(range(2))>0:
                cX+=random.choice(range(0,20))
        else:
            pass

    return bimg, imgBoxes, textboxs,fontType

def setrorate(im):
    # 旋转
    temp = randNum(-1, 1)
    methods=[Image.NEAREST,Image.BILINEAR,Image.BICUBIC]
    if temp != 0:
        #修改为只旋转1-2度
        angle = np.pi / random.randint(90,180) / temp
        if not random.randint(0,2):
            expand=True
        else:
            expand=False
        im = im.rotate(angle,resample=np.random.choice(methods),expand=expand)
    return im

def erode(im):
    #腐蚀
    img = np.array(im)
    kernel = np.ones((3,3),np.uint8)
    img = cv2.erode(img,kernel,iterations = 1)
    return Image.fromarray(img)

def dilate(im):
    #膨胀
    img = np.array(im)
    kernel = np.ones((3,3),np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    return Image.fromarray(img)

def setdim(im):
    # 模糊处理
    fts = [ImageFilter.DETAIL, ImageFilter.SMOOTH, ImageFilter.EDGE_ENHANCE,ImageFilter.SHARPEN]
    return im.filter(np.random.choice(fts))

def enhance(im):
    temp=random.randint(0,3)
    if temp==0:
        im=ImageEnhance.Brightness(im).enhance(random.uniform(0.6,1.4))
    if temp==1:
        im=ImageEnhance.Contrast(im).enhance(random.uniform(0.8,1.3))
    if temp==2:
        im=ImageEnhance.Sharpness(im).enhance(random.uniform(0.5,1.5))
    return im
def perspective(img):
    w = img.size[0]
    h = img.size[1]
    img=np.array(img)
    # 得到透射变换矩阵
    src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    dst0=np.array([[w * random.uniform(0,0.05), h * random.uniform(0,0.05)], 
                    [w * random.uniform(0.95,1.0), h * random.uniform(0.0,0.05)], 
                    [w * random.uniform(0.95,1.0), h * random.uniform(0.9,0.95)], 
                    [w * random.uniform(0,0.05), h * random.uniform(0.95,1.0)]], dtype=np.float32)
    dst = np.array([[w * random.uniform(0,0.1), h * random.uniform(0,0.05)], 
                    [w * random.uniform(0.85,1.0), h * random.uniform(0.05,0.1)], 
                    [w * random.uniform(0.85,1.0), h * random.uniform(0.9,0.95)], 
                    [w * random.uniform(0,0.1), h * random.uniform(0.95,1.0)]], dtype=np.float32)
    if random.randint(0,1):
        transform_matrix = cv2.getPerspectiveTransform(src, dst)
    else:
        transform_matrix = cv2.getPerspectiveTransform(src, dst0)
    # 透射变换完成变形
    img = cv2.warpPerspective(img, transform_matrix, (w, h))
    img=Image.fromarray(img)
    return img

def sp_noise(image,prob=0.02):
    '''
    添加椒盐噪声
    prob:噪声比例 
    '''
    image=np.array(image)
    output = np.zeros(image.shape,np.uint8)
    prob=random.uniform(0.01,0.02)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return Image.fromarray(output)


def gasuss_noise(image, mean=0, var=0.001):
    ''' 
        添加高斯噪声
        mean : 均值 
        var : 方差
    '''
    var=random.uniform(0.001,0.002)
    image=np.array(image)
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)
    #cv.imshow("gasuss", out)
    return Image.fromarray(out)

def imgscreate(file):
    global imgsnum
    # 图像生成和对应文本保存
    im, imgBoxes, texts,fontType = draw_txt()
    font_name=fontType.split("/")[-1]
    for index, box in enumerate(imgBoxes):
        #print (box)
        imgsnum += 1
        if box[0]<0:
            box[0]=0
            box[2]=box[2]-box[0]      
        smimg = im.crop(box)
        name="1"+str(uuid1().__str__())
        #print(name)
        path = os.path.join(root, name)
        if not random.randint(0,2):
            smimg=setdim(smimg)
        if not random.randint(0,2):
            smimg=setrorate(smimg)
            
        if random.randint(0,1):
            smimg=enhance(smimg)
        
        methods=[Image.NEAREST,Image.BILINEAR,Image.BICUBIC,Image.ANTIALIAS]
        smimg = smimg.resize((128, 32), np.random.choice(methods))
        sming = smimg.convert("L")
        sming.save(path+'_'+font_name+texts[index]+".jpg")
        #print(index)
        file.write(name+".jpg "+texts[index]+"\n")
        #print(index)

        if imgsnum % shownum == 0:
            print("create %d picture" % imgsnum)



if __name__ == "__main__":
    start = time.time()
    if not os.path.exists(root):
        os.makedirs(root)
    file=open("image_list.txt","w")
    for i in range(batch):
        imgscreate(file)
    file.close()
    '''
    with open("Statistics.txt", 'w') as fs:
        Stat = ""
        for key, value in wordclassnum.items():
            Stat += key.encode('utf-8') + "->" + str(value) + "\n"
        fs.write(Stat)
    '''
    end = time.time()
    print("spend times:%f" % (end - start))
