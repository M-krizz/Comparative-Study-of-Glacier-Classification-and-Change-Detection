from PIL import Image, ImageChops 
  
img1 = Image.open("c:/Users/Administrator/Desktop/frame_000.jpg") 
img2 = Image.open("c:/Users/Administrator/Desktop/frame_001.jpg") 
  
diff = ImageChops.difference(img1, img2) 

diff.show()