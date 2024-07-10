import argparse
import cv2
import numpy as np
import glob
from mmengine import ProgressBar
from mmseg.apis import inference_model, init_model

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Video and Image Inference Script')
    parser.add_argument('--video', help='Video file path')
    parser.add_argument('--image', help='Path to the directory containing images')
    parser.add_argument('--camera', type=int, help='Camera source (e.g., 0 for default webcam)')
    parser.add_argument('--config', default='configs/pidnet/pidnet-s_2xb6-120k_1024x1024-cityscapes.py', help='Config file path')
    parser.add_argument('--checkpoint', default='checkpoints/pidnet-s_2xb6-120k_1024x1024-cityscapes_20230302_191700-bb8e3bcc.pth', help='Checkpoint file path')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference (e.g., "cpu", "cuda:0")')
    parser.add_argument('--out', type=str, help='Output directory for images or video file path')
    parser.add_argument('--show', action='store_true', help='Show video or images during processing')
    parser.add_argument('--wait-time', type=float, default=0.001, help='Interval of show (seconds), 0 means blocking, default = 0.001')
    args = parser.parse_args()
    return args

def get_images(directory):
    """
    Loads images from the specified directory.
    """
    print("[Console] Accessing folder")
    image_paths = glob.glob(directory)
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
            images.append((image_path, image))
    print(f"[INFO] Loaded {len(images)} image(s)")
    return images

def apply_mask(image, mask, palette):
    """Apply the mask to the image using vectorized operations."""
    colored_mask = palette[mask].astype(np.uint8)
    return cv2.addWeighted(image, 0.5, colored_mask, 0.5, 0)

def process_video(args, model, palette):
    """
    Process the video file for inference.
    """
    # Open the video file
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("[DEBUG] Error: Could not open video.")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[DEBUG] Video is loaded with width={width}, height={height}, fps={fps}")

    # Initialize video writer if output path is specified
    video_writer = None
    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(args.out, fourcc, fps, (width, height))

    # Initialize progress bar
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = ProgressBar(task_num=frame_count)

    # Pre-allocate memory for colored_mask
    colored_mask = np.zeros((height, width, 3), dtype=np.uint8)

    # Create window for display if necessary
    if args.show:
        cv2.namedWindow('Segmented Result', cv2.WINDOW_NORMAL)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform inference on the frame
        result = inference_model(model, frame)
        
        # Apply mask to the frame
        seg = result.pred_sem_seg.data[0].cpu().numpy().astype(np.uint8)
        colored_mask[:] = palette[seg]
        segmented_frame = cv2.addWeighted(frame, 0.5, colored_mask, 0.5, 0)
        
        # Show the video if specified
        if args.show:
            cv2.imshow('Segmented Result', segmented_frame)
            if cv2.waitKey(int(args.wait_time * 1000)) & 0xFF == ord('q'):
                break
        
        # Write the frame to the output video file if specified
        if args.out:
            video_writer.write(segmented_frame)

        # Update the progress bar
        progress_bar.update()

    # Release the video capture and writer
    cap.release()
    if video_writer:
        video_writer.release()
    
    # Destroy all OpenCV windows
    cv2.destroyAllWindows()

def process_images(args, model, palette):
    """
    Process the images for inference.
    """
    images = get_images(args.image)
    count = 1
    for image_path, image in images:
        # Perform inference on the image
        result = inference_model(model, image)

        # Apply mask to the image
        seg = result.pred_sem_seg.data[0].cpu().numpy().astype(np.uint8)
        segmented_image = apply_mask(image, seg, palette)

        # Show the image if specified
        if args.show:
            cv2.imshow('Segmented Result', segmented_image)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break

        # Save the result if output path is specified
        if args.out:
            output_path = f'{args.out}/result_{count}.png'
            cv2.imwrite(output_path, segmented_image)
        count += 1

    # Destroy all OpenCV windows
    if args.show:
        cv2.destroyAllWindows()

def process_camera(args, model, palette):
    """
    Process the live video feed from the camera for inference.
    """
    # Open the camera source
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("[DEBUG] Error: Could not open camera.")
        return

    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[DEBUG] Camera is loaded with width={width}, height={height}, fps={fps}")
    
    # Pre-allocate memory for colored_mask
    colored_mask = np.zeros((height, width, 3), dtype=np.uint8)

    # Create window for display if necessary
    if args.show:
        cv2.namedWindow('Segmented Result', cv2.WINDOW_NORMAL)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform inference on the frame
        result = inference_model(model, frame)

        # Apply mask to the frame
        seg = result.pred_sem_seg.data[0].cpu().numpy().astype(np.uint8)
        colored_mask[:] = palette[seg]
        segmented_frame = cv2.addWeighted(frame, 0.5, colored_mask, 0.5, 0)

        # Show the video if specified
        if args.show:
            cv2.imshow('Segmented Result', segmented_frame)
            if cv2.waitKey(int(args.wait_time * 1000)) & 0xFF == ord('q'):
                break

    # Release the camera
    cap.release()
    
    # Destroy all OpenCV windows
    cv2.destroyAllWindows()

def main():
    """
    Main function to perform video, image, or camera inference.
    """
    args = parse_args()
    
    # Ensure at least one input method is specified
    assert args.video or args.image or args.camera is not None, (
        'Please specify at least one input source (video/images/camera) with the argument "--video", "--images", or "--camera"'
    )

    # Initialize the model
    model = init_model(args.config, args.checkpoint, device=args.device)
    
    # Get dataset meta information
    palette = np.array(model.dataset_meta['palette'])
    classes = model.dataset_meta['classes']
    print(f"[DEBUG] Imported these classes: {classes}")

    if args.video:
        process_video(args, model, palette)
    elif args.image:
        process_images(args, model, palette)
    elif args.camera is not None:
        process_camera(args, model, palette)

if __name__ == '__main__':
    main()