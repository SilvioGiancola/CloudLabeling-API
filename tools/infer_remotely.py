
# import requests
import json
import cv2
import numpy as np
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
# import gdown
from cloudlabeling import cloudlabeling

parser = ArgumentParser(
    description='python tools/infer_remotely.py --image_path=tools/sample_striga.jpg --output_path=tools/sample_striga_out.jpg --project_id=Striga_Strat1',
    formatter_class=ArgumentDefaultsHelpFormatter)

# Advanced Settigns
parser.add_argument("--image_path", type=str, help="path of an image to infer")                   
parser.add_argument("--output_path", type=str, help="path to save the output image")                   
parser.add_argument("--project_id", type=str, default="MSCOCO", help="ID of the project for inference")                   

# Advanced Settigns
parser.add_argument("--HOST", type=str, default="cloudlabeling.org", help="host IP")                        
parser.add_argument("--PORT", type=int, default=4000, help="commmunication port")
parser.add_argument("--device", type=str, default="cpu", help="ID of the project for inference")                   
args = parser.parse_args()

# create CloudLabeling instance
cloud_labeler = cloudlabeling.CloudLabeling(HOST=args.HOST, PORT=args.PORT, device=args.device)

# inference 
results = cloud_labeler.infer_remotely(args.image_path, args.project_id)

# display results
if results["error"] is not None:
    print("Error in inference:")
    print(results["error"])
    print(results)


image = cloud_labeler.display_BB(cv2.imread(args.image_path), results)        
cv2.imwrite(args.output_path, image)

print(results)

# for result in results["detection"]:
#     print(result)

