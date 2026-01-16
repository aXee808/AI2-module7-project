from fastapi import FastAPI
import pandas as pd


# uncomment all to use minio (docker-compose)
# boto3 library
"""
import boto3 
from dotenv import load_dotenv
import os
"""

# Load .env environment variables
"""
load_dotenv()
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION_NAME = os.getenv("AWS_REGION_NAME")
AWS_ENDPOINT_URL = os.getenv("AWS_ENDPOINT_URL")
AWS_BUCKET_MEDIA = os.getenv("AWS_BUCKET_MEDIA")
"""
# boto3
"""
s3client = boto3.resource(
    service_name="s3",
    endpoint_url=AWS_ENDPOINT_URL,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    aws_session_token=None,
    config=boto3.session.Config(signature_version="s3v4"),
    region_name=AWS_REGION_NAME,
    verify=False,
)
"""

# copy csv file from minio S3 bucket to local container
#s3client.Bucket(AWS_BUCKET_MEDIA,).download_file("titanic.csv", "./data/titanic.csv")

# declare API
app = FastAPI()
print("API loading !")

# load local csv file
df_titanic = pd.read_csv("./data/titanic.csv")

# endpoint to check api availability
@app.get("/health")
async def health():
    return {"status":"ok"}

# endpoint to query api (titanic passengers)
@app.get("/passengers")
async def get_passengers(age_min : int,age_max : int):
    """
    Passengers endpoint : return survived & dead passengers (male,female)

    params : age_min (integer) minimum age range
           : age_max (integer) maximum age range 

    return : (json) {'dead_female': integer,'dead_male': integer,'survived_female':integer,'survived_male':integer}
    """

    # select passengers rows between minimum age and maximum age
    df_select = df_titanic.loc[(df_titanic['Age']>= age_min)&(df_titanic['Age']<=age_max) ,['Survived','Sex']]

    # group passengers by survived status and sex, with count rows
    df_agg = df_select.groupby(['Survived','Sex']).agg({'Survived': ['count']}).reset_index()

    # change columns name
    df_agg.columns= ['a','b','c']

    # extract & format to json the result
    result = df_agg['c'].to_list()
    jsonstr = {'dead_female':result[0],'dead_male':result[1],'survived_female':result[2],'survived_male':result[3]}

    # return string json format
    return jsonstr
