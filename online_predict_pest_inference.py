from google.cloud import automl
from datetime import datetime
from operator import itemgetter
import google.cloud.aiplatform as aip
from psql_info import *

from google.cloud import aiplatform
from google.cloud.aiplatform.gapic.schema import predict
from google.protobuf.json_format import MessageToDict

from google.cloud import storage
from google.cloud import bigquery
import numpy as np
import base64
import json
import io
from PIL import Image, ImageDraw
import pandas as pd
import os
import json
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text as sqlalchemy_text

PROJECT_ID = 'semios-imagery'
REGION = 'us-central1'

def get_psql_engine(user, password, host, database):
    print(user, password, host, database)
    return create_engine('postgresql://{}:{}@{}:5432/{}'.format(
            user,
            password,
            host,
            database
        ))
def get_previous_detections_from_pg(engine, image_id):
    query = '''
      SELECT *
      FROM trapcoords
      WHERE trapimage_id = {} 
    '''.format(image_id)
    with engine.connect() as conn:
        df = pd.read_sql(sqlalchemy_text(query), con=conn)
    return df

def get_images_to_label_pg(engine, trap_id, start_date=None, end_date=None, number_of_days = 5):

    if (start_date == None) and (end_date == None):
        today = datetime.today()
        end_date = (today + timedelta(days=2)).strftime("%Y-%m-%d")
        start_date = (today - timedelta(days=number_of_days)).strftime("%Y-%m-%d")
    
    query = \
    '''
      SELECT id, hash, DATE(taken_on) AS date_taken_on, taken_on
      FROM trapimages 
      WHERE COALESCE(trap_mac, trap_id) = '{}'
        AND taken_on BETWEEN '{}' AND '{}'
      ORDER BY taken_on
    '''.format(trap_id, start_date, end_date)
    print(f'THIS IS THE QUERY: {query}')
    with engine.connect() as conn:
        df = pd.read_sql(sqlalchemy_text(query), con=conn)
    return df

def get_images_to_label(project_id, trap_id, start_date=None, end_date=None, number_of_days = 5):

    if (start_date == None) and (end_date == None):
        today = datetime.today()
        end_date = (today + timedelta(days=2)).strftime("%Y-%m-%d")
        start_date = (today - timedelta(days=number_of_days)).strftime("%Y-%m-%d")
    
    client = bigquery.Client(project_id)
    query = \
    '''
      SELECT t.id, t.hash, t.taken_on
      FROM `semios-bq-prod.semios_rds_public_views.trapimages` t
      WHERE COALESCE(trap_mac, trap_id) = '{}'
        AND taken_on BETWEEN '{}' AND '{}'
      ORDER BY taken_on
    '''.format(trap_id, start_date, end_date)

    return client.query(query).to_dataframe()

def get_maintenance_labels_from_simulation_table(engine, image_hashes):

    image_hashes_str = ','.join(["'" + str(elem) + "'" for elem in image_hashes])
    query = '''
      SELECT hash, maintenance
      FROM trapimages 
      WHERE hash IN ({})
    '''.format(image_hashes_str)
    print(query)
    with engine.connect() as conn:
        df = pd.read_sql(sqlalchemy_text(query), con=conn)
    return df


def get_relabeled_damage_and_obstruction_images(project_id):

    client = bigquery.Client(project_id)
    query = \
    '''
      SELECT 
        image_hash
      FROM `semios-imagery.dev_soverduin.training_dataset_for_review_damage_or_obstruction`
    '''
    print(query)

    return client.query(query).to_dataframe()


def get_previous_detections(project_id, image_id):

    client = bigquery.Client(project_id)
    query = \
    '''
      SELECT *
      FROM `semios-bq-prod.semios_rds_public_views.trapcoords` t
      WHERE trapimage_id = {} 
    '''.format(image_id)

    return client.query(query).to_dataframe()

def predict_image_object_detection_sample(
    project: str,
    endpoint_id: str,
    encoded_content: str,
    confidence_threshold: float = 0.05,
    location: str = "us-central1",
    api_endpoint: str = "us-central1-aiplatform.googleapis.com",
):
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple
    # requests.
    client = aiplatform.gapic.PredictionServiceClient(
        client_options=client_options
    )
    instance = predict.instance.ImageObjectDetectionPredictionInstance(
        content=encoded_content,
    ).to_value()
    instances = [instance]
    # See gs://google-cloud-aiplatform/schema/predict/params/image_object_detection_1.0.0.yaml
    # for the format of the parameters.
    parameters = predict.params.ImageObjectDetectionPredictionParams(
        confidence_threshold=confidence_threshold, max_predictions=1000,
    ).to_value()
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )

    return response


def run_pest_detector_model(photo, endpoint_id, project_id, location, confidence_threshold):
    
    response = predict_image_object_detection_sample(
       project=project_id,
       endpoint_id=endpoint_id,
       encoded_content=photo,
       confidence_threshold=confidence_threshold,
       location=location
    )
    payload = response.predictions[0]
    insect_acros = MessageToDict(payload._pb['displayNames'])
    boxes = MessageToDict(payload._pb['bboxes'])
    scores = MessageToDict(payload._pb['confidences'])
    payload_dict = {
        'boxes': boxes,
        'scores': scores,
        'insect_acros': insect_acros
    }
    return (payload_dict)


def predict_automl(
    project_id: str,
    model_id: str,
    photo: any,
    location: str = "us-central1",
    score_threshold: str = "0.5"
):
    automl_client = automl.PredictionServiceClient()
    base64_bytes = photo.encode('ascii')
    img_bytes = base64.b64decode(base64_bytes)
    image = automl.Image(image_bytes=img_bytes)
    payload = automl.ExamplePayload(image=image)
    params = {'score_threshold': score_threshold}
    model_full_id = automl.AutoMlClient.model_path(project_id, location, model_id)
    request = automl.PredictRequest(name=model_full_id, payload=payload, params=params)
    response = automl_client.predict(request=request)

    return response

def parse_object_detection_payload(payload):

    xmin = payload.image_object_detection.bounding_box.normalized_vertices[0].x
    ymin = payload.image_object_detection.bounding_box.normalized_vertices[0].y
    xmax = payload.image_object_detection.bounding_box.normalized_vertices[1].x
    ymax = payload.image_object_detection.bounding_box.normalized_vertices[1].y

    box = [ymin, xmin, ymax, xmax]
    label = payload.display_name
    score = payload.image_object_detection.score

    return (score, label, box)


def parse_classification_payload(payload):

    label = payload.display_name
    score = payload.classification.score

    return (score, label)


def run_liner_issue_detection(
    project_id: str,
    hashed_bucket_name: str,
    meta_bucket_name: str,
    photo
):
    """ Run Liner Detection """

    client = bigquery.Client(project_id)
    gcs_client = storage.Client(project=project_id)
    hashed_bucket = gcs_client.get_bucket(hashed_bucket_name)
    meta_bucket = gcs_client.get_bucket(meta_bucket_name)

    try:
        model_name = 'detect_liner_issue'
        model_id = 'ICN6310292944876208128'
        score_threshold = '0.0'
        model_type = 'classification'
        model_info = {
                      'model_name': model_name,
                      'model_id': model_id,
                      'score_threshold': score_threshold,
                      'model_type': model_type
                     }
        project_id = 'semios-imagery'
        location = 'us-central1'
        print('STARTING PREDICTION...')
        response = predict_automl(
            project_id=project_id,
            model_id=model_id,
            photo=photo,
            location=location,
            score_threshold=score_threshold
        )
        labels_scores = []
        if len(response.payload) == 0:
            if model_type == 'object_detection':
                labels_scores = [('no_object_detected', None, None)]
            else:
                print(
                "No labels detected for {} with threshold {}!".format(model_name, score_threshold))

        for r in response.payload:
            if model_type == 'classification':
                score, label = parse_classification_payload(r)
                labels_scores.append((label, score))
            elif model_type =='object_detection':
                score, label, box = parse_object_detection_payload(r)
                labels_scores.append((label, score, box))
            else:
                print(
                "Not classification or object detection results")
                return

        labels_scores = sorted(labels_scores, key=itemgetter(1), reverse=True)
    except Exception as e:
        print('GOT HERE',e)
    return labels_scores

def get_img_bytes_from_json(data):
    photo = data.get('photo')
    base64_bytes = photo.encode('ascii')
    return (photo, base64.b64decode(base64_bytes))

def get_img_from_blob(hash_, meta_bucket):
    blob = meta_bucket.get_blob('{}.json'.format(hash_))
    if blob is None:
        print('{} meta does not exist...'.format(hash_))
        return None
    content = blob.download_as_string()
    data = json.loads(content)
    photo, img_bytes = get_img_bytes_from_json(data)
    img = Image.open(io.BytesIO(img_bytes))
    return (photo, img)

def get_jpg_from_blob(hash_, hashed_bucket):
    blob = hashed_bucket.get_blob('{}.jpg'.format(hash_))
    if blob is None:
        print('{} jpg does not exist...'.format(hash_))
        return None
    content = blob.download_as_string()
    img = Image.open(io.BytesIO(content)) 
    return (img)


meta_bucket_name = 'semios-prod-trapimages-meta'
hashed_bucket_name = 'semios-prod-trapimages-hashed'
gcs_client = storage.Client(project=PROJECT_ID)
meta_bucket = gcs_client.get_bucket(meta_bucket_name)
print(meta_bucket)
hashed_bucket = gcs_client.get_bucket(hashed_bucket_name)
location='us-central1'

project_id = 'moth-recognition'

user = PSQL_USER 
host = HOST_PROD_REPLICA 
password = PASSWORD_PROD_REPLICA 
database = DATABASE_PROD_REPLICA 
engine_replica = get_psql_engine(user, password, host, database)

print('GOT HERE!!')
#engine = get_psql_engine(user, password, host, database)

database = DATABASE_PROD_READ_REPLICA 
host = HOST_PROD_READ_REPLICA 
password = PASSWORD_PROD_READ_REPLICA 
engine = get_psql_engine(user, password, host, database)
print('got engine',engine)



df_image_hashes = get_relabeled_damage_and_obstruction_images(project_id)


trap_id = '001C2C1B260CB36C'
start_date = '2023-10-08' #None
end_date = '2023-10-11' #None
print('engine',engine)
df = get_images_to_label_pg(engine_replica, trap_id, start_date=start_date, end_date=end_date, number_of_days=2)
print(df)
df = df[
         (df['id'] == 11820582)
#         | (df['id'] == 10357018)
#        (df['id'] == 10261020) | 
#        (df['id'] == 10257991) | 
#        (df['id'] == 10257588) | 
#        (df['id'] == 10257503) |
#        (df['id'] == 10257345) |
#        (df['id'] == 10256589) 
         ]
print(df)
print(df['hash'])
print(df['id'])
from PIL import ImageFont
lst = []
for i, row in df.iterrows():
    image_id = row['id']
    hash_ = row['hash']
    taken_on = row['date_taken_on']
    if trap_id.startswith('A'):
        hash_ = f"{trap_id}/{row['date_taken_on']}/{hash_}"
        print(hash_)
    print(image_id, hash_)
    photo, img = get_img_from_blob(hash_,  meta_bucket)
    img_arr = np.array(img)
    print(img_arr.shape)
    df_pg = get_previous_detections_from_pg(engine, image_id)
    img1 = ImageDraw.Draw(img)
    print('img size', img.size)
    print(df_pg)

    print('number of catches',len(df_pg))
    # Draw boxes currently in trapcoords
    for j, jrow in df_pg.iterrows():
        br = (jrow['br_x'], jrow['br_y'])
        tl = (jrow['tl_x'], jrow['tl_y'])
        parent_id = jrow['parent_trapcoord_id']
        print('br',br)
        if parent_id > 0:
            img1.rectangle([br, tl], width=5, outline='cyan')
        else:
   #         print('hey!! here: ',br, tl)
            img1.rectangle([br, tl], width=4, outline='magenta')


    labels_scores = run_pest_detector_model(photo, '8646295558039797760', project_id, location, 0.0001)
    # Draw ML boxes
    for label, score, coords in zip(labels_scores['insect_acros'], labels_scores['scores'], labels_scores['boxes']):
        xmin, xmax, ymin, ymax =  coords[0], coords[1], coords[2], coords[3]
        if score > 0.01:
                lst.append([trap_id, image_id, taken_on, label, score, xmin, xmax, ymin, ymax])
        bottom_left = (int(xmin * img.size[0]), int(ymin*img.size[1]))  
        top_right = (int(xmax * img.size[0]), int(ymax*img.size[1]))  
        draw = ImageDraw.Draw(img) 
        print(label,score)
        if score > 0.01:
            #print('LABEL',label)
            if (label == 'CM') and (0.5 < score < 0.9):
               print('HERE',label, score) #, coords)
               img1.rectangle([bottom_left, top_right], width=3,outline='cyan')
               draw.text((xmin*img.size[0],ymin*img.size[1]-10), label + ': ' + str(np.round(score,3)), (0,255,255))
            elif (label == 'CM') and (score > 0.9) :
               print('HERE',label, score) #, coords)
               img1.rectangle([bottom_left, top_right], width=3,outline='orange')
               draw.text((xmin*img.size[0],ymin*img.size[1]-10), label + ': ' + str(np.round(score,3)), (0,155,255))
            elif (label == 'CM' and score > 0.1):
               img1.rectangle([bottom_left, top_right], width=3,outline='blue')
               draw.text((xmin*img.size[0],ymax*img.size[1]+1), label + ': ' + str(np.round(score,3)), (0,0,255))
            elif label == 'CM':
                img1.rectangle([bottom_left, top_right], width=3,outline='red')
                draw.text((xmin*img.size[0],ymax*img.size[1]+1), label + ': ' + str(np.round(score,3)), (0,0,255))
            else:
                img1.rectangle([bottom_left, top_right], width=2,outline='orange')
                draw.text((xmin*img.size[0],ymax*img.size[1]+1), label + ': ' + str(np.round(score,3)), (0,0,255))
    img.show()
