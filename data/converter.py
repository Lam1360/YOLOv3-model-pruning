import scipy.io as sio
from PIL import Image
import os, glob
import datetime
import shutil

running_from_path = os.getcwd()
created_images_dir = 'images'
created_labels_dir = 'labels'
data_dir = 'data'   # data_dir为脚本所在的文件夹

def hms_string(sec_elapsed):    # 格式化显示已消耗时间
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60.
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)

def generate_dir(set_name, root_path):   # 往images和labels文件夹下生成相应的文件夹
    images_dir = os.path.join(root_path, 'images')
    annotation_dir = os.path.join(root_path, 'annotations')

    new_images_dir = os.path.join(created_images_dir, set_name)   # 将图片从原来的文件夹复制到该文件夹下
    new_annotation_dir = os.path.join(created_labels_dir, set_name)

    if not os.path.exists(new_images_dir):
        os.makedirs(new_images_dir)

    if not os.path.exists(new_annotation_dir):
        os.makedirs(new_annotation_dir)

    for img in glob.glob(os.path.join(images_dir, "*.jpg")):    # 将图片从原来的文件夹复制到新文件夹下
        shutil.copy(img, new_images_dir)

    os.chdir(annotation_dir)        # 切换到annotation的路径下
    matlab_annotations = glob.glob("*.mat")  # 仅仅包含文件名，不包含路径
    os.chdir(running_from_path)     # 切换回原来的路径

    for matfile in matlab_annotations:
        filename = matfile.split(".")[0]

        pil_image = Image.open(os.path.join(images_dir, filename+".jpg"))

        content = sio.loadmat(os.path.join(annotation_dir, matfile), matlab_compatible=False)

        boxes = content["boxes"]

        width, height = pil_image.size

        with open(os.path.join(new_annotation_dir, filename+".txt"), "w") as hs:
            for box_idx, box in enumerate(boxes.T):
                a = box[0][0][0][0]
                b = box[0][0][0][1]
                c = box[0][0][0][2]
                d = box[0][0][0][3]

                aXY = (a[0][1], a[0][0])
                bXY = (b[0][1], b[0][0])
                cXY = (c[0][1], c[0][0])
                dXY = (d[0][1], d[0][0])

                maxX = max(aXY[0], bXY[0], cXY[0], dXY[0])
                minX = min(aXY[0], bXY[0], cXY[0], dXY[0])
                maxY = max(aXY[1], bXY[1], cXY[1], dXY[1])
                minY = min(aXY[1], bXY[1], cXY[1], dXY[1])

                # clip,防止超出边界
                maxX = min(maxX, width-1)
                minX = max(minX, 0)
                maxY = min(maxY, height-1)
                minY = max(minY, 0)

                # (<absolute_x> / <image_width>)
                norm_width = (maxX - minX) / width

                # (<absolute_y> / <image_height>)
                norm_height = (maxY - minY) / height

                center_x, center_y = (maxX + minX) / 2, (maxY + minY) / 2

                norm_center_x = center_x / width
                norm_center_y = center_y / height

                if box_idx != 0:
                    hs.write("\n")

                hs.write("0 %f %f %f %f" % (norm_center_x, norm_center_y, norm_width, norm_height)) # 0表示类别

def create_txt(dirlist, filename):
    with open(filename, "w") as txtfile:   # 在data文件夹下生成txt文件
        imglist = []

        for dir in dirlist:     # dir='images/test'
            imglist.extend(glob.glob(os.path.join(dir, "*.jpg")))   # img='images/test/abc.jpg'

        for idx, img in enumerate(imglist):
            if idx != 0:
                txtfile.write("\n")
            txtfile.write(os.path.join(data_dir, img))    # 加上前缀data

if __name__ == '__main__':
    start_time = datetime.datetime.now()

    generate_dir("train", "hand_dataset/training_dataset/training_data")    # 第一个参数表示生成的文件夹的名称
    generate_dir("test", "hand_dataset/test_dataset/test_data")
    generate_dir("validation", "hand_dataset/validation_dataset/validation_data")

    create_txt((os.path.join(created_images_dir, 'train'),          # 将train和validation文件夹下的图片合并成train
                os.path.join(created_images_dir, 'validation')),
               'train.txt')
    create_txt((os.path.join(created_images_dir, 'test'), ),
               'valid.txt')

    end_time = datetime.datetime.now()
    seconds_elapsed = (end_time - start_time).total_seconds()
    print("It took {} to execute this".format(hms_string(seconds_elapsed)))