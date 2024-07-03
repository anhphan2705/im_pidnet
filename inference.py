from mmseg.apis import init_model, inference_model, show_result_pyplot

config_path = 'configs\pidnet\pidnet-s_2xb6-120k_1024x1024-cityscapes.py'
checkpoint_path = 'checkpoints\pidnet-s_9591acc_7691iou.pth'
img_path = 'samples/image.png'


model = init_model(config_path, checkpoint_path)
result = inference_model(model, img_path)

vis_iamge = show_result_pyplot(model, img_path, result,with_labels=False, out_file='samples/output/result.png')