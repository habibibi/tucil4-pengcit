@echo off

@REM Download dataset
curl -L -o data/vehicle-type-recognition.zip  https://www.kaggle.com/api/v1/datasets/download/kaggleashwin/vehicle-type-recognition
unzip -o data/vehicle-type-recognition.zip -d data

@REM move folders
move data/Dataset/Bus data/Bus
move data/Dataset/Car data/Car
move data/Dataset/motorcycle data/motorcycle
move data/Dataset/Truck data/Truck

@REM clean up files
rmdir /s /q data\Dataset
del data\vehicle-type-recognition.zip

@REM Training model
python train_SVM.py
