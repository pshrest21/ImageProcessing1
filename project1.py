from PIL import Image
import numpy as np
#import cv2

#reading P2 type .pgm files
def readpgm(name, new_image_name):

    with open(name) as f:
         lines = f.readlines()

    # Ignores commented lines
    for l in list(lines):
        if l[0] == '#':
            lines.remove(l)

    # Makes sure it is ASCII format (P2)
    assert lines[0].strip() == 'P2' 

    # Converts data to a list of integers
    data = []
    for line in lines[1:]:
        data.extend([int(c) for c in line.split()])

    img_array = np.array(data[3:])
    new_img = np.zeros([512,512])

    n = 512
    final = [img_array[i * n:(i + 1) * n] for i in range((len(img_array) + n - 1) // n )]  #final is 512x512 array 
     
    for i in range(0,512):
        for j in range(0,512):
            new_img[i][j] = final[i][j]

    img = Image.fromarray(new_img)
    img.convert('L').save(new_image_name+'.png', optimize = True) 
    return new_img

#Function to load the array with the data from the given file
#This is only for reading P5 type .pgm files
def loadPixels(pgmf):
    pgmf.readline()
    pgmf.readline()

    (width, height) = [int(i) for i in pgmf.readline().split()]
    depth = int(pgmf.readline())
    assert depth <= 255
    
    raster = []
    for y in range(height):
        row = []
        for y in range(width):
            row.append(ord(pgmf.read(1)))
        raster.append(row)      
    return raster
    

#Function to display the image from the list we have obtained from above function
def displayImage(img_name, pgmf):
    img = Image.new("I", (512, 512))
    pixels = []
    pixels = img.load()
    raster = []
    raster = loadPixels(pgmf)

    for i in range(img.size[0]):
        for j in range(img.size[1]):           
            pixels[i, j] = raster[i][j] * 255

     
    img.save(img_name + '.png')
    return raster
    
def customSpatResolution(img_array,x):
    new_array = np.zeros([x, x])
    for i in range(0, len(img_array), 2):
        for j in range(0, len(img_array[i]), 2):        
            new_array[i][j] = (img_array[i][j] + img_array[i][j+1] + img_array[i+1][j] + img_array[i+1][j+1])/4
       
    return new_array[::2, 0::2] 
    #return new_array  

def getResolution(img_array,x):  
    new_array = np.zeros([x , x])
    new_array[::2, ::2] = img_array
    for i in range(0, len(new_array), 2):
        for j in range(0, len(new_array[i]), 2):        
            new_array[i][j+1] = new_array[i+1][j] = new_array[i+1][j+1] = new_array[i][j]

    return new_array    
 
def mainSpatResolution(img_array,x,y,new_image_name):
    n = int(x / y)
    
    if(n == 1):
        #img_array = img_array.transpose()
        img = Image.fromarray(img_array)
        img.convert('L').save(new_image_name+'.png', optimize = True)
        return
    
    new_image = customSpatResolution(img_array,x)
    #new_image = getResolution(new_image, 512)
    #new_image = new_image.transpose()


    counter = 2
    if(n > 2):
        while (counter < n):
            size = (new_image.size) ** (1/2)
            new_image = customSpatResolution(new_image, int(size))
            #new_image = getResolution(new_image, )
            counter = counter * 2

    counter = 1
    while(counter < n):
        size = ((new_image.size) ** (1/2)) * 2
        new_image = getResolution(new_image, int(size))
        counter = counter * 2 

    img = Image.fromarray(new_image)
    img.convert('L').save(new_image_name+'.png', optimize = True) 
    return new_image
    
def redBitsPerPixel(img_array, x, y, new_image_name): #x = original grayscale and y = new grayscale 
    #y = 8 - y 
    n = (2**x)/(2**y)
    for i in range(0, len(img_array)):
        for j in range(0, len(img_array)):
            if(img_array[i][j] >= (2**x)/2):

                img_array[i][j] = int(img_array[i][j] / n)
                img_array[i][j] = (img_array[i][j] * n) + (n - 1)  
            else:
                img_array[i][j] = int(img_array[i][j] / n) * n
                
        # if(img_array[i][j] >= (2**y)):            
        #     if (img_array[i][j] >= 128):
        #         img_array[i][j] = img_array[i][j] + (2**y)
        #     else:
        #         img_array[i][j] = abs(img_array[i][j] - (2**y)) 

                

    img = Image.fromarray(img_array)
    #img = img.transpose(Image.FLIP_LEFT_RIGHT)
    #img = img.transpose(Image.ROTATE_90)
    img.convert('L').save(new_image_name+'.png', optimize = True) 
    #return img_array


#Operations on fish.pgm
pgmf = open('fish.pgm', 'rb')
img_fish = np.array(displayImage('fish', pgmf)) #convert byte image to array image
# img_fish = readImageFile('fish.pgm', 'fish')
# print(img_fish[0])
# print(len(img_fish[0]))

mainSpatResolution(img_fish, 512, 256, 'fish_256') #reduce spatial resolution to 256x256
mainSpatResolution(img_fish, 512, 128, 'fish_128') #reduce spatial resolution to 128x128
mainSpatResolution(img_fish, 512, 64, 'fish_64') #reduce spatial resolution to 64x64

redBitsPerPixel(img_fish, 8, 6, 'fish_6_bits_pp') #reduce bits per pixel from 8 to 6

reducedGrayLevelImage = mainSpatResolution(img_fish, 512, 128, 'fish_128_4_bits_pp') #reduce the 6 bits per pixel image's spatial resolution to 128x128 
redBitsPerPixel(reducedGrayLevelImage, 6, 4, 'fish_128_4_bits_pp')

#Operations on modern.pgm
img_modern = readpgm('modern.pgm', 'modern')
mainSpatResolution(img_modern, 512, 256, 'modern_256') #reduce spatial resolution to 256x256
mainSpatResolution(img_modern, 512, 128, 'modern_128') #reduce spatial resolution to 128x128
mainSpatResolution(img_modern, 512, 64, 'modern_64') #reduce spatial resolution to 64x64

redBitsPerPixel(img_modern, 8, 6, 'modern_6_bits_pp') #reduce bits per pixel from 8 to 6

reducedGrayLevelImage = mainSpatResolution(img_modern, 512, 128, 'modern_128_4_bits_pp') #reduce the 6 bits per pixel image's spatial resolution to 128x128 

redBitsPerPixel(reducedGrayLevelImage, 6, 4, 'modern_128_4_bits_pp')

#Operations on jet.pgm
img_jet = readpgm('jet.pgm', 'jet')
mainSpatResolution(img_jet, 512, 256, 'jet_256') #reduce spatial resolution to 256x256
mainSpatResolution(img_jet, 512, 128, 'jet_128') #reduce spatial resolution to 128x128
mainSpatResolution(img_jet, 512, 64, 'jet_64') #reduce spatial resolution to 64x64

redBitsPerPixel(img_jet, 8, 6, 'jet_6_bits_pp') #reduce bits per pixel from 8 to 6

reducedGrayLevelImage = mainSpatResolution(img_jet, 512, 128, 'jet_128_4_bits_pp') #reduce the 6 bits per pixel image's spatial resolution to 128x128 

redBitsPerPixel(reducedGrayLevelImage, 6, 4, 'jet_128_4_bits_pp')



'''
def redSpatResolution(img_array, x, y, new_image_name):
    w,h = img_array.shape
    p,q = (x,y)
    
    resize = img_fish.reshape(h//p,p,w//q,q).mean((1,3), keepdims = 1)
    newImage = np.repeat(np.repeat(resize,(x),axis = (1)),(q),axis=3).reshape(img_fish.shape) 
    
    #newImage = np.rot90(newImage)    
    #newImage = newImage.transpose()
    
    img = Image.fromarray(newImage)
    img.convert('L').save(new_image_name+'.png', optimize = True) 


img_modern = readpgm('modern.pgm', 'modern')
im = Image.open('modern.png')
im1 = im.point(lambda x: int(x/128)* 128)
im1.save('newImage1.png')

im2 = im.point(lambda x: int(x/64)* 64)
im2.save('newImage2.png')

im3 = im.point(lambda x: int(x/32)* 32)
im3.save('newImage3.png')

im4 = im.point(lambda x: int(x/16)* 16)
im4.save('newImage4.png')

im5 = im.point(lambda x: int(x/8)* 8)
im5.save('newImage5.png')

im6 = im.point(lambda x: int(x/4)* 4)
im6.save('newImage6.png')

im7 = im.point(lambda x: int(x/2)* 2)
im7.save('newImage7.png')


def readImageFile(name): #could have used this but I had to be extra for no reason. Regreting now...
    new_img = cv2.imread(name, -1)      
    #img = Image.fromarray(new_img)
    #img.convert('L').save(image_name+'.png', optimize = True) 
    return new_img

img_modern = readpgm('jet.pgm', 'modern')
#pgmf = open('jet.pgm', 'rb')
#img_modern = np.array(displayImage('fish', pgmf))
redBitsPerPixel(img_modern, 8, 7, 'modern_7_bits_pp')
redBitsPerPixel(img_modern, 8, 6, 'modern_6_bits_pp')
redBitsPerPixel(img_modern, 8, 5, 'modern_5_bits_pp')
redBitsPerPixel(img_modern, 8, 4, 'modern_4_bits_pp')
redBitsPerPixel(img_modern, 8, 3, 'modern_3_bits_pp')
redBitsPerPixel(img_modern, 8, 2, 'modern_2_bits_pp')
redBitsPerPixel(img_modern, 8, 1, 'modern_1_bits_pp')

'''




















    