from mmseg.apis import init_model, inference_model, show_result_pyplot
import cv2
import glob

config_path = 'configs\pidnet\pidnet-s_2xb6-120k_1024x1024-cityscapes.py'
checkpoint_path = 'checkpoints\pidnet-s_9591acc_7691iou.pth'
img_path = 'samples/*'

def get_images(directory=img_path):
    """
    Loads images from the specified directory.

    Args:
        directory (str): Path to the directory containing the images.

    Returns:
        List of loaded images (list of numpy.ndarray).

    Raises:
        InvalidDirectoryError: If the directory is invalid or no images are found.
    """
    try:
        print("[Console] Accessing folder")
        image_paths = glob.glob(directory)
        print(image_paths)
        if len(image_paths) == 0:
            raise FileExistsError("[ERROR] Invalid directory or no images found.")
        images = []
        # Add images to memory
        print("[Console] Loading Images")
        for image_path in image_paths:
            image = cv2.imread(image_path)
            if image is None:
                print(f"[ERROR] Unable to load image: {image_path}")
            else:
                images.append(image)
        print(f"[INFO] Loaded {len(images)} image(s)")
        return images
    except Exception as e:
        print(str(e))
        raise e


model = init_model(config_path, checkpoint_path)
images = get_images()

count=1
for image in images:
    result = inference_model(model, image)
    vis_iamge = show_result_pyplot(model, image, result, with_labels=True, out_file=f'samples/output/result_{count}.png')
    count+=1