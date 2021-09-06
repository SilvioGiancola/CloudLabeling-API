import requests
import json
import cv2
import numpy as np
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
# import gdown

class CloudLabeling:
    def __init__(self, HOST="cloudlabeling.org", PORT=4000, device="cpu"):
        self.HOST = HOST                
        self.PORT = PORT                
        self.device = device


    def infer_remotely(self, image_path, project_id="MSCOCO", request_type="image/jpeg"):
        # send request
        if request_type == "image/jpeg":
            r = requests.post(f'http://{self.HOST}:{self.PORT}/api/predict', 
                            data=open(image_path, 'rb').read(), 
                            headers={"content-type":"image/jpeg",
                                     "project_id": project_id,
                                     "device": self.device})

        elif request_type == "gdrive/jpeg":
            r = requests.post(f'http://{self.HOST}:{self.PORT}/api/predict', 
                            headers={"content-type":"gdrive/jpeg",
                                     "gdrive/jpeg":image_path,
                                     "project_id": project_id, 
                                     "device": self.device})

        elif request_type == "gdrive/mp4":
            r = requests.post(f'http://{self.HOST}:{self.PORT}/api/predict', 
                            headers={"content-type":"gdrive/mp4",
                                     "gdrive/mp4":image_path,
                                     "project_id": project_id, 
                                     "device": self.device})
        # return dictionary with {"boxes", "labels_words", "scores"}
        # print(r.text)

        results = json.loads(r.text)
        # if error in inference
        if results["error"] is not None:
            print("Error in inference:")
            print(results["error"])
            print(results)
            return {"error": results["error"], "detection": [], "labels": []}
        
        # Arrange results
        new_results = []
        for box, label, score in zip(results["boxes"], results["labels_words"], results["scores"]):
            new_results.append({"box":box, "label":label, "confidence":score}) 
        # list of labels
        labels =list(set([r["label"] for r in new_results]))

        new_results = {"detection": new_results, "labels": labels, "error": results["error"]}
        return new_results

    def display_BB(self, image, results):
        # labels =list(set(results["labels_words"]))
        random.seed(10) # seed to have reproducible shuffling
        color = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(results["labels"]))] 

        # for boxes, labels_words, scores in zip(results["boxes"], results["labels_words"], results["scores"]):
        for result in results["detection"]:
            boxes, labels_words, scores = result["box"], result["label"], result["confidence"]
            # extract the bounding box coordinates
            (x1, y1) = (int(boxes[0]), int(boxes[1])) # top left corner
            (x2, y2) = (int(boxes[2]), int(boxes[3])) # bottom right corner
            # assign color
            col = color[results["labels"].index(labels_words)]
            # draw a bounding box rectangle and label on the image
            cv2.rectangle(image, (x1, y1), (x2, y2), col)
            text = "{}: {:.4f}".format(labels_words, scores)
            cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, col, 2)
        return image

