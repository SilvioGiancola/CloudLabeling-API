
# import requests
import json
import cv2
import numpy as np
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import gdown
from cloudlabeling import cloudlabeling

parser = ArgumentParser(
    description='Example to request detection to CloudLabeling',
    formatter_class=ArgumentDefaultsHelpFormatter)

# Advanced Settigns
parser.add_argument("--image_path", type=str, help="path of an image to infer")                   
parser.add_argument("--output_path", type=str, help="path to save the output image")                   
parser.add_argument("--project_id", type=str, default="MSCOCO", help="ID of the project for inference")                   

# Advanced Settigns
parser.add_argument("--HOST", type=str, default="cloudlabeling.org", help="host IP")                        
parser.add_argument("--PORT", type=int, default=4000, help="commmunication port")
parser.add_argument("--device", type=str, default="cpu", help="ID of the project for inference")                   
parser.add_argument("--request_type", type=str, default="image/jpeg", help="ID of the project for inference")                   
args = parser.parse_args()


# python cloudlabeling.py --image_path=sample_striga.jpg --output_path=sample_striga_out.jpg --project_id="Striga_Strat1" --device="cuda:0" --HOST=10.68.74.8 --PORT 4001
# python cloudlabeling.py --image_path=https://drive.google.com/uc?id=1DK6iUAbHte-KlzGjZTKjeaGB1OCjI7_z --output_path=sample_striga_out.jpg --project_id="Striga_Strat1" --device="cuda:0" --request_type="gdrive/jpeg" --HOST=10.68.74.8 --PORT 4001
# python cloudlabeling.py --image_path=https://drive.google.com/uc?id=1odxr1EwBl-A8SWGBgf31mk5uTC-rCAZG --output_path=sample_striga_out.mp4 --project_id="Striga_Strat1" --device="cuda:0" --request_type="gdrive/mp4" --HOST=10.68.74.8 --PORT 4001


# create CloudLabeling instance
cloud_labeler = cloudlabeling.CloudLabeling(HOST=args.HOST, PORT=args.PORT, device=args.device)

# inference 
results = cloud_labeler.infer_remotely(args.image_path, args.project_id, args.request_type)



# display results
if results["error"] is not None:
    print("Error in inference:")
    print(results["error"])
    print(results)

elif args.request_type == "image/jpeg":
    for box, label, score in zip(results["boxes"], results["labels_words"], results["scores"]):
        print(f"Found {label} in position {box} with confidence {score}")
    
    image = cloud_labeler.display_BB(cv2.imread(args.image_path), results)        
    cv2.imwrite(args.output_path, image)

elif args.request_type == "gdrive/jpeg":
    for box, label, score in zip(results["boxes"], results["labels_words"], results["scores"]):
        print(f"Found {label} in position {box} with confidence {score}")
    
    output = '/tmp/gdrive_image_query.jpg'
    gdown.download(args.image_path, output, quiet=True)
    args.image_path = output

    image = cloud_labeler.display_BB(cv2.imread(args.image_path), results)        
    cv2.imwrite(args.output_path, image)

elif args.request_type == "gdrive/mp4":
    # for i_frame, result in enumerate(results):
    #     print(f"Frame {i_frame}")
    #     for box, label, score in zip(result["boxes"], result["labels_words"], result["scores"]):
    #         print(f"Found {label} in position {box} with confidence {score}")

    print(len(results))
    with open(args.output_path+'.txt', 'w') as outfile:
        json.dump(results, outfile,indent=4)


    output = '/tmp/gdrive_video_query.mp4'
    gdown.download(args.image_path, output, quiet=True)
    # args.image_path = output

    video_input = cv2.VideoCapture(output)

    cnt=0
    success, image = video_input.read()
    height, width, layers = image.shape
    size = (width,height)       
    print(size)
    out = cv2.VideoWriter(args.output_path,cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
    # results=[]
    while success:
        # print(cnt)
        # if cnt % 30 == 0:
        # results.append(infer_image(image))
        success, image = video_input.read() 
        res = results[cnt]
        image = cloud_labeler.display_BB(image, res)
        out.write(image)               
        cnt+=1
    out.release()
    video_input.release()
    cv2.destroyAllWindows()
