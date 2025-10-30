from ultralytics import SAM

model = SAM('mobile_sam.pt') 

model.predict('img_1.jpg', save=True, show=True)