import argparse
import cv2
from mmengine import ProgressBar
from mmseg.apis import inference_model, init_model, show_result_pyplot

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Video Inference Script')
    parser.add_argument('--video', default='samples/video.mp4', help='Video file path')
    parser.add_argument('--config', default='configs/pidnet/pidnet-s_2xb6-120k_1024x1024-cityscapes.py', help='Config file path')
    parser.add_argument('--checkpoint', default='checkpoints/pidnet-s_2xb6-120k_1024x1024-cityscapes_20230302_191700-bb8e3bcc.pth', help='Checkpoint file path')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference (e.g., "cpu", "cuda:0")')
    parser.add_argument('--out', type=str, help='Output video file path')
    parser.add_argument('--show', action='store_true', help='Show video during processing')
    parser.add_argument('--wait-time', type=float, default=1, help='Interval of show (seconds), 0 means blocking')
    args = parser.parse_args()
    return args

def main():
    """
    Main function to perform video inference.
    """
    args = parse_args()
    
    # Ensure at least one output method is specified
    assert args.out or args.show, (
        'Please specify at least one operation (save/show the video) with the argument "--out" or "--show"'
    )

    # Initialize the model
    model = init_model(args.config, args.checkpoint, device=args.device)

    # Open the video file
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video is loaded with width={width}, height={height}, fps={fps}")

    # Initialize video writer if output path is specified
    video_writer = None
    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(args.out, fourcc, fps, (width, height))

    # Initialize progress bar
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = ProgressBar(task_num=frame_count)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform inference on the frame
        result = inference_model(model, frame)
        
        # Visualize the result without displaying
        frame_with_result = show_result_pyplot(model, frame, result, with_labels=False, show=False)
        
        # Show the video if specified
        if args.show:
            cv2.imshow('video', frame_with_result)
            if cv2.waitKey(args.wait_time) & 0xFF == ord('q'):
                break
        
        # Write the frame to the output video file if specified
        if args.out:
            video_writer.write(frame_with_result)

        # Update the progress bar
        progress_bar.update()

    # Release the video capture and writer
    cap.release()
    if video_writer:
        video_writer.release()
    
    # Destroy all OpenCV windows
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()