:github_url: https://github.com/SilvioGiancola/CloudLabeling

.. role:: raw-html(raw)
   :format: html
.. default-role:: raw-html

API Calls
================

1. API call for training
-----------------


2. API call for inference
-----------------

Example for inference::

  curl -H "Content-Type: image/jpeg" \
  -H "project_id: MSCOCO" \
  -X POST \
  --data-binary @/path/to/your/image.jpg \
  http://cloudlabeling.org:4000/api/predict


