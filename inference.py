from mmseg.apis import MMSegInferencer

# Load models into memory
inferencer = MMSegInferencer(model='pidnet-s_2xb6-120k_1024x1024-cityscapes', device='cuda:0')
# Inference
inferencer('image.png', show=True)