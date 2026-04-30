from ultralytics import YOLO

# 1. 학습된 내 모델 불러오기 (경로 수정 필요)
model = YOLO("best-7.pt")

# 2. 모델에게 영상/이미지 보여주고 예측 시키기
# save=True 옵션이 네모 박스가 그려진 결과물을 파일로 저장해 줍니다.
results = model.predict(source="test_detect.mp4", conf=0.15, imgsz=1280, save=True)

print("테스트 완료! runs/detect/predict 폴더를 확인해보세요.")