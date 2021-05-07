from PIL import Image
start_x = 102
start_y = 96
change_x = 77
change_y = 77
image = Image.open('image_list.png')
for i in range(8):
    start_x = 102
    for j in range(8):
        crop = image.crop((start_x, start_y, start_x+change_x, start_y+change_y))
        
        crop.save(str(i)+str(j)+".png",'PNG')
        start_x += change_x
    start_y+=change_y