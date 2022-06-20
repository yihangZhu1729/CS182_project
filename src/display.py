from utils import get_img, display_image
import numpy as np

def img_display(data_batch_id: int):
    """
    display a random image in the given data_batch
    """
    random_idx = np.random.randint(10000)
    img_label, img_data = get_img(data_batch_id, random_idx)
    saved_img = display_image(img_data)
    # saved_img.save("data/task1.png")
    print("label:", img_label)
    return (img_label, img_data)


if __name__ == '__main__':
    img_label, img_data = img_display(2)