# Cloudlabeling API

```bash
conda create -n cloudlabeling python pip
conda activate cloudlabeling
pip install cloudlabeling
```

## How to use (Python)

```python
from cloudlabeling import cloudlabeling

cloud_labeler = cloudlabeling.CloudLabeling()

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
