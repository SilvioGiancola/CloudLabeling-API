from flask import Flask, request, jsonify, send_from_directory

import os
import sys
from shutil import copyfile
import numpy as np
import json
import cv2

from mmdet.apis import init_detector, inference_detector
import mmcv
# import gdown

        
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import time
import boto3
import botocore
import requests

# import face_recognition

def draw_Boxes(frame, boxes, ext="JPEG"):

    # WITH PIL
    # with Image.open(frame_path) as im:

        # font = ImageFont.truetype('DejaVuSansMono.ttf', size=20)

    # draw = ImageDraw.Draw(im)
    for rect, score, label in zip(boxes["boxes"], boxes["scores"], boxes["labels_words"]):
        if score > 0.7:
            # print(rect)
            frame = cv2.rectangle(frame, (int(rect[0]), int(rect[1])), (int(rect[2]), int(rect[3])), (0, 128, 0), 5)
            frame = cv2.putText(frame, label, (int(rect[0]), int(rect[1]+10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2, cv2.LINE_AA)

    return frame
        # write to stdout
        # im.save(frame_path, ext)


class FlaskServer(object):
    def __init__(self, HOST, PORT):
        self.model = None
        self.PORT = PORT
        self.HOST = HOST
        self.model = None         
        self.initialize_model("Striga_Strat1")

        self.face_model = None


    def initialize_model(self, project_id, device=None):
        # Initialize the model
        # args: project [string]: Path of the triaing project to open
        # return the error code (0: no error, 1: model does not exist locally, 2: model does not exsit on s3)

        print("Initialize Model")


        # check if model exist locally
        task_id=100
        for file in ["faster_rcnn_r50_fpn_1x_coco.py", "latest.pth"]:

            file_fullpath = f"training/{project_id}/{file}"
            os.makedirs(os.path.dirname(file_fullpath), exist_ok=True)
            if not os.path.exists(file_fullpath):
                print(f"{file_fullpath} does not exist locally")


                for task_id in range(task_id, 0, -1):
                    # print(task_id)
                    # try:
                        
                    url = f'http://cloudlabeling.org:4000/training/{project_id}/{task_id}/{file}'
                    head = requests.head(url)
                    # print(url, head)

                    if head.status_code == 200:
                        r = requests.get(url)

                        with open(file_fullpath, 'wb') as f:
                            f.write(r.content)

                        # Retrieve HTTP meta-data
                        # print(r.status_code)
                        # print(r.headers['content-type'])
                        # print(r.encoding)
                        

                        # http://cloudlabeling.org:4000/training/{project_id}/{task_id}/latest.pth
                        # http://cloudlabeling.org:4000/training/{project_id}/{task_id}/faster_rcnn_r50_fpn_1x_coco.py
                        # http://cloudlabeling.org:4000/training/Fish/10/faster_rcnn_r50_fpn_1x_coco.py
                        # faster_rcnn_r50_fpn_1x_coco

                        break
                    # except:
                    #     continue


            
            # list_objects = s3_client.list_objects_v2(Bucket=Config.BUCKET_NAME)
            
            # if "Contents" not in list_objects:
            #     error = "local model not found: s3 is empty!"
            #     print(error)
            #     return error

            # projects_in_s3 = [f["Key"].split("/")[0] for f in  list_objects["Contents"]]
            # projects_in_s3 = list(set(projects_in_s3))

            # print("projects_in_s3", projects_in_s3)
            # if project_id not in projects_in_s3:
            #     error = "local model not found: project not found on s3!"
            #     print(error)
            #     return error

            # list_objects = s3_client.list_objects_v2(Bucket=Config.BUCKET_NAME, Prefix=project_id)
            # tasks_in_s3 = [f["Key"].split(project_id + "/")[1].split("/")[0] for f in  list_objects["Contents"]]
            
            # print("tasks_in_s3", tasks_in_s3)
                
            # last_task_in_s3 = sorted(list(set(tasks_in_s3)))[-1]

            # # create local folder for model
            # local_folder = f"training/{project_id}/{last_task_in_s3}"
            # os.makedirs(local_folder)

            # # download configuration
            # with open(f"{local_folder}/faster_rcnn_r50_fpn_1x_coco.py", "wb") as dest_file: 
            #     s3_client.download_fileobj(Config.BUCKET_NAME,
            #                                 f"{project_id}/{last_task_in_s3}/faster_rcnn_r50_fpn_1x_coco.py", 
            #                                 dest_file)

            # # download checkpoint
            # with open(f"{local_folder}/latest.pth", "wb") as dest_file: 
            #     s3_client.download_fileobj(Config.BUCKET_NAME,
            #                                 f"{project_id}/{last_task_in_s3}/latest.pth", 
            #                                 dest_file)
        

        # identify last training task available locally  
        # task_id = "" #sorted(list(map(int, os.listdir(f"training/{project_id}/"))))[-1]
        print(f"Loading project_id = {project_id}")

        # local config and checkpoint file    
        config_file = f"training/{project_id}/faster_rcnn_r50_fpn_1x_coco.py"
        checkpoint_file = f'training/{project_id}/latest.pth'

        # if config and checkpoint file are missing, raise error 
        if not os.path.exists(config_file) or not os.path.exists(checkpoint_file):
            return "local model not found"
            

        # clean symlink if present
        # if os.path.islink(checkpoint_file):
        #     print("symlink detected!")
        #     real_file = f'training/{project_id}/{task_id}/{os.readlink(checkpoint_file)}'
        #     os.remove(checkpoint_file) # remove symlink
        #     copyfile(real_file, checkpoint_file) # copy real file

        # initialize detector
        self.project_id = project_id
        self.device = device
        self.config = mmcv.Config.fromfile(config_file)
        self.classes = self.config.data.train.classes
        # build the model from a config file and a checkpoint file
        if device is None:
            device = "cuda"
        try:
            self.model = init_detector(config_file, checkpoint_file, device=device)            
        except:
            self.model = init_detector(config_file, checkpoint_file, device="cpu")
        # # backup the config and checkpoint to s3
        # config_file_s3 = f"{project_id}/{task_id}/faster_rcnn_r50_fpn_1x_coco.py"
        # checkpoint_file_s3 = f'{project_id}/{task_id}/latest.pth'

        # def key_exists(client, mykey, mybucket):
        #     response = client.list_objects_v2(Bucket=mybucket, Prefix=mykey)
        #     if response:
        #         for obj in response.get('Contents', []):
        #             if mykey == obj['Key']:
        #                 return True
        #     return False

        # for file_type, file_localpath, file_s3path in [["config", config_file, config_file_s3], ["ckpt", checkpoint_file, checkpoint_file_s3]]:
        #     # check if key is backed up
        #     if not key_exists(s3_client, file_s3path, Config.BUCKET_NAME):
        #         print(f"{file_type} does not exist on s3!", file_localpath)
        #         with open(file_localpath,"rb") as f:
        #             s3_client.upload_fileobj(f,Config.BUCKET_NAME,file_s3path)
        
        return None
    
    def infer_image(self, image):

        # compute prediction
        result = inference_detector(self.model, image)

        # import numpy as np
        bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)

        score_thr = 0.3
        if score_thr > 0:
            assert bboxes.shape[1] == 5
            scores = bboxes[:, -1]
            inds = scores > score_thr
            scores = bboxes[inds, -1]
            boxes = bboxes[inds, :4]
            labels = labels[inds]

        scores = scores.tolist()
        boxes = boxes.tolist()
        labels = labels.tolist()
        # print("scores:", scores)
        # print("bboxes:", boxes)
        # print("labels", labels)

        labels_words = [str(self.classes[idx]) for idx in labels]
        # print(labels_words)
        return {
                'boxes': boxes,
                'labels_words': labels_words,
                'labels_idx': labels,
                'scores': scores
                }

    def get_index(self):

        HTML = "<html>"
        HTML += f"<head><title>List of Trained Projects</title></head>"
        HTML = "<body>"

        for project in sorted(os.listdir("training")):

            HTML += f"""<details>
                            <summary><strong>Project ID: {project}</strong></summary>"""
            for task in sorted(os.listdir(os.path.join("training",project))):

                logjson = [logjson for logjson in os.listdir(os.path.join("training",project,task)) if logjson.endswith(".log.json")]
                if len(logjson) == 0:
                    HTML += f"""<details>
                            <summary>Task ID: {task}</summary>
                            <ul>
                            <li>epoch: N/A</li>
                            <li>mAP: N/A</li>
                            <li><a href="training/{project}/{task}/latest.pth">Model</a></li>
                            </ul>
                        </details>
                        """
                    continue
                logjson = os.path.join("training",project,task,sorted(logjson)[-1])
                with open(logjson) as f:
                    lines = f.readlines()
                data = json.loads(lines[-1])
                HTML += f"""<details>
                            <summary>Task ID: {task}</summary>
                            <ul>
                            <li>epoch: {data['epoch']}</li>
                            <li>mAP: {data['mAP']}</li>
                            <li><a href="training/{project}/{task}/epoch_50.pth">Model</a></li>
                            </ul>
                        </details>
                        """
                    
            HTML += f'<p></p></details>'
        HTML += f'</article>'
        HTML += "</body>"
        HTML += "</html>"
        return HTML

    def download(self, filename):   

        # sever.download(filename)
        print(f"Downloading {filename}")
        return send_from_directory(directory='training', filename=filename)

    def predict(self, request):   
        content_type = request.headers.get('content-type')
        project_id = request.headers.get('project_id')
        task = request.headers.get('task')
        device = request.headers.get('device')

        answer = {"error": None} # default no error
        if self.model is None or not project_id == self.project_id or not device == self.device:
            error = self.initialize_model(project_id, device)
            if error is not None:
                answer["error"] = error
            
        ## back-compatilibilty
        if (content_type == "image/jpeg" and task is None):
            # Read request
            image_bytes = request.data
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if answer["error"] is None:
                results = self.infer_image(image)
                answer.update(results)
            print("answer", answer)
            return jsonify(answer)
        
        elif (content_type == "video/mp4" and task is None):
            # Read request

            output_format = request.headers.get('return')
            image_bytes = request.data
            file = open("tmp.mp4", "wb")
            file.write(image_bytes)
            file.close()
            # nparr = np.frombuffer(image_bytes, np.uint8)
            # image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if answer["error"] is None:
                cap = cv2.VideoCapture("tmp.mp4")
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                print(fps, (w, h))
                if output_format == "video":
                    out = cv2.VideoWriter("out.mp4", cv2.VideoWriter_fourcc('M','P','4','V'), fps, (w,h))
                results_video = {}
                i =0 
                while(True):
                    # Capture frame-by-frame
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # results = {}
                    frame_drop = request.headers.get('frame_drop')
                    if frame_drop is None:
                        frame_drop = 10
                    if i % int(frame_drop) == 0:
                        results = self.infer_image(frame)
                    print({str(i): results})
                    results_video.update({str(i): results})
                    i = i + 1

                    if output_format == "video":
                        #TODO: Draw BB
                        # p
                        draw_Boxes(frame,results)
                        out.write(frame)
                    # print(results_video)
                # results = {"inference_video":"TBD"}
                answer.update(results_video)
            cap.release()
            os.remove("tmp.mp4")
            # print("answer", answer)
            # os.remove("tmp.mp4")
            # return jsonify(answer)
            if output_format == "video":
                out.release()
                file = open("out.mp4", "rb")
                byte = file.read()
                file.close()
                os.remove("out.mp4")
                return byte

            else:
                return jsonify(answer)


            # print(f"Downloading tmp.mp4")
            # return send_from_directory(directory='.', filename="tmp.mp4")
            # return 


        ## back-compatilibilty
        elif (content_type == "image/jpeg" and task is "detect/face"):
            # Read request
            image_bytes = request.data
            nparr = np.frombuffer(image_bytes,np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if answer["error"] is None:
                results = self.detect_face(image)    
                answer.update(results)    
            # print("answer", answer)
            return jsonify(answer)

        # elif (content_type == "gdrive/jpeg"):
        #     url = request.headers.get('gdrive/jpeg')

        #     output = '/tmp/gdrive_image.jpg'
        #     gdown.download(url, output, quiet=True)
        #     image = cv2.imread(output)
            
        #     if answer["error"] is None:
        #         results = self.infer_image(image)    
        #         answer.update(results)   
        #     return jsonify(answer)

        # elif (content_type == "gdrive/mp4"):
        #     url = request.headers.get('gdrive/mp4')

        #     results=[]
        #     if answer["error"] is None:

        #         output = '/tmp/gdrive_video.mp4'
        #         gdown.download(url, output, quiet=True)
        #         cap = cv2.VideoCapture(output)
        #         cnt=0
        #         success, image = cap.read()
        #         while success:
        #             # print(cnt)
        #             # if cnt % 30 == 0:
        #             results.append(self.infer_image(image))
        #             success, image = cap.read() 
        #             cnt+=1
        #         cap.release()
        #         cv2.destroyAllWindows()

        #     answer.update(results)  
        #     return jsonify(answer)

        else:
            return jsonify(answer)



# TODO: Add API to infer video from client
# TODO: save models on s3 buckets after training is done
# TODO: fetch model on s3 in inference if not available in folder
# TODO: save last and best model only, no shortcut for last (update mmdet/apis?)
# TODO: investigate torch2ONNX converters
