import os
import numpy as np
import itertools
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2
path_in = "F:/UBC-OCEAN/train_thumbnails"
path_out = "F:/UBC-OCEAN/global_images"

img_path = list(
                                [os.path.join(path_in, img)
                            for img in os.listdir(path_in)
                            if (os.path.isfile(os.path.join(path_in,
                            img)) and img.endswith('png'))]
                            )
# paths_in = list(os.path.join(path_in, cls_fol) for cls_fol in os.listdir(path_in))
# img_fol_dir = list(itertools.chain.from_iterable([[os.path.join(cls_fol, img_fol)
#                             for img_fol in os.listdir(cls_fol)  
#                             if (os.path.isdir(os.path.join(cls_fol, img_fol))
#                             )] for cls_fol in paths_in]))

# img_path = list(itertools.chain.from_iterable([[os.path.join(img_fol, img)
#                             for img in os.listdir(img_fol)  
#                             if (os.path.isfile(os.path.join(img_fol, img))
#                             and img.endswith('.png') and not("_" in img))] for img_fol in img_fol_dir]))


# print(img_path)
for img in img_path:
    print(img)
    image = cv2.imread(img)
    img_out = cv2.resize(image, (224, 224))
    cv2.imwrite(os.path.join(path_out, os.path.basename(img).split("_")[0]+"_global.png")
, img_out) 