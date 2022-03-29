# cloudlabeling API
API call for cloudlabeling.org

## How to install (pip)

```bash
conda create -n cloudlabeling python pip
conda activate cloudlabeling
pip install cloudlabeling
```

## How to use (Python)

```python
from cloudlabeling import cloudlabeling

api_token = "..."
cloud_labeler = cloudlabeling.CloudLabeling(api_token=api_token)

image_path = "tools/sample_striga.jpg"
results = cloud_labeler.infer_remotely(image_path, project_id="MSCOCO")
```

Results output in JSON format

```json
{
   "detection":[ # list of detections
      {
         "box":[
            268.44647216796875, # x min
            4.61001443862915,   # y min
            2401.08740234375,   # x max
            1919.837646484375   # y max
         ],
         "label":"bowl",        # class name
         "label_idx":45,        # class ID
         "confidence":0.7302282 # confidence score
      }, ...
   ],
   "labels":[ # list of labels in detection
      "dining table",
      "bowl",
      "cake"
   ],
   "error":None # error (if any)
}
```

## cURL inference (command line)

```bash
curl -H "Content-Type: image/jpeg" \
-H "project_id: MSCOCO" \
-H "device: cuda:0" \
-H "api_token: xxx" \
-X POST \
--data-binary @/path/to/image.jpg \
http://cloudlabeling.org:4000/api/predict
```

e.g. :

```
curl -H "Content-Type: image/jpeg" -H "project_id: MSCOCO" -H "device: cuda:0" -X POST --data-binary @/Users/giancos/Desktop/proj/image001.jpg http://cloudlabeling.org:4000/api/predict
```

## TODO list

- [x] define format for results
- [x] release tools to public
- [ ] handle images on gdrive
- [ ] handle images in numpy/TF/PT instead of image path

## Update pip:

python setup.py upload


## Environemnt for server

conda create -y -n CloudLabeling python=3.7
conda activate CloudLabeling
conda install -y pytorch=1.6 torchvision=0.7 cudatoolkit=10.1 -c pytorch
conda install requests=2.25.1
pip install Flask opencv-python
pip install mmcv-full==1.2.4 mmdet==2.11.0
pip install gdown
pip install boto3

## CloudLabeling Server

python tools/run_server.py

python tools/infer_remotely.py --image_path=tools/sample_striga.jpg --output_path=tools/sample_striga_out.jpg --project_id=Striga_Strat1 --HOST localhost


## Sanity check

curl -H "Content-Type: image/jpeg" -H "project_id: Striga_Strat1" -H "device: cuda:0" -H "api_token: 303630fcc6a04793ba7e09fc0336a037" -X POST --data-binary @/Users/giancos/Desktop/proj_csv/image001.jpg http://10.68.74.28:4000/api/predict                             

curl -H "Content-Type: image/jpeg" -H "project_id: 75" -H "device: cuda:0" -H "api_token: 303630fcc6a04793ba7e09fc0336a037" -X POST --data-binary @/Users/giancos/Desktop/WhiteHelmet/000015.jpg http://10.68.74.28:4000/api/predict

curl -H "Content-Type: image/jpeg" -H "project_id: 77" -H "device: cuda:0" -H "api_token: 303630fcc6a04793ba7e09fc0336a037" -X POST --data-binary @/Users/giancos/Desktop/proj_csv/image001.jpg http://10.68.74.28:4000/api/predict 



curl -H "Content-Type: image/jpeg" -H "project_id: Striga_Strat1" -H "device: cuda:0" -H "api_token: 303630fcc6a04793ba7e09fc0336a037" -X POST --data-binary @/Users/giancos/Desktop/proj_csv/image001.jpg http://cloudlabeling.org:4000/api/predict    
curl -H "Content-Type: image/jpeg" -H "project_id: Striga_Strat2" -H "device: cuda:0" -H "api_token: 303630fcc6a04793ba7e09fc0336a037" -X POST --data-binary @/Users/giancos/Desktop/proj_csv/image001.jpg http://cloudlabeling.org:4000/api/predict                             

curl -H "Content-Type: image/jpeg" -H "project_id: 75" -H "device: cuda:0" -H "api_token: 303630fcc6a04793ba7e09fc0336a037" -X POST --data-binary @/Users/giancos/Desktop/WhiteHelmet/000015.jpg http://cloudlabeling.org:4000/api/predict

curl -H "Content-Type: image/jpeg" -H "project_id: 77" -H "device: cuda:0" -H "api_token: 303630fcc6a04793ba7e09fc0336a037" -X POST --data-binary @/Users/giancos/Desktop/proj_csv/image001.jpg http://cloudlabeling.org:4000/api/predict 



# LARGE FILES
curl -H "Content-Type: image/jpeg" -H "project_id: 75" -H "device: cuda:0" -H "api_token: 303630fcc6a04793ba7e09fc0336a037" -X POST --data-binary @/Users/giancos/Downloads/360ImagesTr/1.jpg http://cloudlabeling.org:4000/api/predict

curl -H "Content-Type: image/jpeg" -H "project_id: 75" -H "device: cuda:0" -H "api_token: 303630fcc6a04793ba7e09fc0336a037" -X POST --data-binary @/Users/giancos/Downloads/360ImagesTr/1.jpg http://10.68.74.28:4000/api/predict


curl https://cloudlabeling-models-product.s3.ap-south-1.amazonaws.com/mar_files/75.mar --output 75.mar 
curl -X POST "http://10.68.74.28:8081/models?url=75.mar&model_name=75&initial_workers=1&synchronous=true"

curl http://10.68.74.28:8081/models

curl http://10.68.74.28:8080/predictions/75 -F "data=@1.jpg"
curl http://10.68.74.28:8080/predictions/75 -T 1.jpg

-T /Users/giancos/Downloads/360ImagesTr/1.jpg