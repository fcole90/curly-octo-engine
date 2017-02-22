import os
from PIL import Image
from statistics import pstdev
import sys


class ImageMerger:
    """
    Merges images together
    """
    def __init__(self, file_list):
        """

        Parameters
        ----------
        file_list: list of str
            list of file paths
        """
        self.image_list = []
        self.width_list = []
        self.height_list = []
        for file in file_list:
            self.image_list.append(Image.open(file))

        self._initialize_dimensions_lists()

    def _initialize_dimensions_lists(self):
        # Creates a list of each width and height for each image
        for img in self.image_list:
            width, height = img.size
            print("Thumb size: " + str(img.size))
            self.width_list.append(width)
            self.height_list.append(height)

    def _get_dimensions_deviations(self):
        # Get a tuple: (deviation of the width, deviation of the height)
        return pstdev(self.width_list), pstdev(self.height_list)

    def define_image_size(self, x_list, y_list):
        x_size = 0
        y_size = y_list[0]
        for x in x_list:
            x_size += x

        for y in y_list:
            if y < y_size:
                y_size = y

        print("Image size: " + str(x_size) + "x" + str(y_size))

        return x_size, y_size

    def merge(self):
        """
        Merge the images

        Merge as a totem if height has an higher deviation,
        as a landscape otherwise. Borders are cut.

        Returns
        -------
        Image:
            the merged image

        """
        wdev, hdev = self._get_dimensions_deviations()
        xstart = 0
        ystart = 0

        i = 0

        # Create a totem
        if wdev <= hdev:
            print("Using totem..")
            height, width = self.define_image_size(self.height_list, self.width_list)
            x_factor = 0
            y_factor = 1

            big_picture = Image.new("RGB", (width, height))

            for img in self.image_list:
                img_w, img_h = img.size
                diff = (img_w - width) / 2
                img.crop((diff, 0, img_w - diff, img_h))
                img.copy()
                big_picture.paste(img, (0, ystart))
                ystart += img_h

                i += 1

        # Create a landscape
        else:
            print("Using landscape..")
            width, height = self.define_image_size(self.width_list, self.height_list)
            x_factor = 1
            y_factor = 0

            big_picture = Image.new("RGB", (width, height))

            for img in self.image_list:
                img_w, img_h = img.size
                diff = (img_h - height) / 2
                img.crop((0, diff, img_w, img_h - diff))
                img.copy()
                big_picture.paste(img, (xstart, 0))
                xstart += img_w

        return big_picture

    def merge_and_save(self, save_dir=None, name=None):
        """
        Does what it says.

        Parameters
        ----------
        save_dir: str
        name: str
        """
        img = self.merge()
        if not save_dir:
            save_dir = os.getcwd()
        if not name:
            name = "bigpic.jpg"
        save_file = os.path.join(save_dir, name)
        img.save(save_file)

def test():
    imglist = []
    for file in os.listdir(os.path.join(os.getcwd(), "0")):
        imglist.append(os.path.join("0", file))

    print(imglist)
    im = ImageMerger(imglist)
    im.merge_and_save()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Missing argument")
    elif sys.argv[1] in ["-t", "--test"]:
        test()
    else:
        for fdir in sys.argv[1:]:
            imglist = []
            for file in os.listdir(fdir):
                imglist.append(os.path.join(fdir, file))

            im = ImageMerger(imglist)
            im.merge_and_save(fdir, fdir + ".jpg")




