# JPG2PNG
from PIL import Image
import cv2 as cv
import os

def JPG2PNG(jpgPath):
    img = cv.imread(jpgPath)
    if img is not None:
        # w, h = img.shape[::-1]
        infile = jpgPath
        outfile = os.path.splitext(infile)[0] + ".png"
        img = Image.open(infile)
        # img = img.resize((int(w / 2), int(h / 2)), Image.ANTIALIAS)
        try:
            if len(img.split()) == 3:
                # prevent IOError: cannot write mode RGBA as BMP
                # r, g, b, a = img.split()
                img = img.convert("RGBA")
                img.save(outfile)
                os.remove(jpgPath)
            # else:
            #     img.convert('RGBA').save(outfile)
            #     os.remove(PngPath)
            return outfile
        except Exception as e:
            print("JPG转换PNG 错误", e)
if __name__ == '__main__':
    for i in range(#图片数量):
        jpg_name=os.path.join("./","{}.jpg".format(i))
        JPG2PNG(jpg_name)


# PNG2JPG

def PNG2JPG(PngPath):
    img = cv.imread(PngPath, 0)
    if img is not None:
        w, h = img.shape[::-1]
        infile = PngPath
        outfile = os.path.splitext(infile)[0] + ".jpg"
        img = Image.open(infile)
        img = img.resize((int(w / 2), int(h / 2)), Image.ANTIALIAS)
        try:
            if len(img.split()) == 4:
                # prevent IOError: cannot write mode RGBA as BMP
                r, g, b, a = img.split()
                img = Image.merge("RGB", (r, g, b))
                img.convert('RGB').save(outfile, quality=70)
                os.remove(PngPath)
            else:
                img.convert('RGB').save(outfile, quality=70)
                os.remove(PngPath)
            return outfile
        except Exception as e:
            print("PNG转换JPG 错误", e)


if __name__ == '__main__':
    for i in range(# 图像数量):
        png_name=os.path.join("./","{}.png".format(i))
        PNG2JPG(png_name)
