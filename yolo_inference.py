from ultralytics import YOLO

model= YOLO('models/best.pt')  

results= model.predict("input_videos/WhatsApp Video 2025-04-13 at 21.55.19_7fe0cb8e.mp4", save=True)
print(results[0])  

print("*************************************************************")

for box in results[0].boxes:
    print(box)
    