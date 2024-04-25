from flask import Flask, request, jsonify
from flask_cors import CORS
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Part,
)
import os
import vertexai
from typing import Dict, List, Union
from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
from dotenv import load_dotenv
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_core.tools import Tool
import json
import re

def predict_custom_trained_model_sample(
    project: str,
    endpoint_id: str,
    instances: Union[Dict, List[Dict]],
    location: str = "us-central1",
    api_endpoint: str = "us-central1-aiplatform.googleapis.com",
):
    """
    `instances` can be either single instance of type dict or a list
    of instances.
    """
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    # The format of each instance should conform to the deployed model's prediction input schema.
    instances = instances if isinstance(instances, list) else [instances]
    instances = [
        json_format.ParseDict(instance_dict, Value()) for instance_dict in instances
    ]
    parameters_dict = {}
    parameters = json_format.ParseDict(parameters_dict, Value())
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )
    # print("response")
    # print(" deployed_model_id:", response.deployed_model_id)
    # html = ""
    # for video in response.predictions:
    #     html += "<video controls>"
    #     html += f'<source src="data:video/mp4;base64,{video}" type="video/mp4">'
    #     html += "</video>"
    return response

def predict_gemini(prompt, project_id, location, model_id="gemini-1.5-pro-preview-0409"):
    vertexai.init(project=project_id, location=location)
    model = GenerativeModel(model_id)

    # Set model parameters
    generation_config = GenerationConfig(
        temperature=1.0,
        top_p=1.0,
        top_k=32,
        candidate_count=1,
        max_output_tokens=8192,
    )

    # Set safety settings
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    }

    # Set contents to send to the model
    contents = [prompt]

    # Prompt the model to generate content
    response = model.generate_content(
        contents,
        generation_config=generation_config,
        safety_settings=safety_settings,
    )

    return response.text

def parse_json_from_gemini(json_str: str):
    """Parses a dictionary from a JSON-like object string.

    Args:
      json_str: A string representing a JSON-like object, e.g.:
        ```json
        {
          "key1": "value1",
          "key2": "value2"
        }
        ```

    Returns:
      A dictionary representing the parsed object, or None if parsing fails.
    """

    try:
        # Remove potential leading/trailing whitespace
        json_str = json_str.strip()

        # Extract JSON content from triple backticks and "json" language specifier
        json_match = re.search(r"```python\s*(.*?)\s*```", json_str, re.DOTALL)

        if json_match:
            json_str = json_match.group(1)

        return json.loads(json_str)
    except (json.JSONDecodeError, AttributeError):
        return None

app = Flask(__name__)
CORS(app)

load_dotenv()
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'vertex_ai_service_account.json'
location = os.environ.get('LOCATION')
endpoint_id = os.environ.get('ENDPOINT_ID')
project_id = os.environ.get('PROJECT_ID')
os.environ["GOOGLE_CSE_ID"] = os.environ.get('GOOGLE_CSE_ID')
os.environ["GOOGLE_API_KEY"] = os.environ.get('GOOGLE_API_KEY')

search = GoogleSearchAPIWrapper()

def top5_results(query):
    return search.results(query, 5)

prompt_template = """
You are an enviromentalist and your job is to help people to view extreme cliamte events from their perspective. 
Perform the following tasks with the contents in the URL and put your answer in a Python dictonary. All answers should be in {} unless specified.
URL: {}
Tasks:
1. (dict key: "summary") Summarize the event in short paragraphs and use local landmarks in {} to compare the scale of the event.
2. (dict key: "ELIF") Explain the cause and impacts of the the event to a 5 years old.
3. (dict key: "image_prompt") Generate 2 detailed image prompts, seperated by comma, that can visualize the extreme cliamte event described below locally. Draft in English and answer just the image prompt.
4. (dict key: "search_prompt") Generate a search prompt so that people can use this to search in Google for more related information.
"""

@app.route('/')
def greetings():
   return 'Welcome to Climate Lens API!'

@app.route('/api/v1/climate_lens')
def gcp_summary_video():
    url = request.args.get('url')
    user_region = request.args.get('user_region')
    user_lang = request.args.get('user_lang')

    response = predict_gemini(prompt_template.format(user_lang, url, user_region), project_id, location)
    response_json = parse_json_from_gemini(response)

    summary = response_json['summary']
    elif_ = response_json['ELIF']
    image_prompt = response_json['image_prompt']
    search_prompt = response_json['search_prompt']

    image_prompts = image_prompt.split(",")
    images = []
    for prompt in image_prompts:
        prediction = predict_custom_trained_model_sample(
            project="611661169313",
            endpoint_id="2514983314956222464",
            location="us-east1",
            instances={
                "prompt": prompt,
                "height": 1024,
                "width": 1024,
                "num_inference_steps": 8,
            },
            api_endpoint = "us-east1-aiplatform.googleapis.com"
        )
        images.append(prediction.predictions[0])
    
    tool = Tool(
        name="Google Search Snippets",
        description="Search Google for recent results.",
        func=top5_results,
    )
    suggested_links = tool.run(search_prompt)
    
    return jsonify(
        summary = summary,
        elif_ = elif_,
        images = images,
        suggested_links = suggested_links
    )

if __name__ == '__main__':
    app.run(debug=True)