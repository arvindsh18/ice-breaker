from flask import Flask, render_template, request, jsonify
from ice_breaker import ice_break
import os

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    AZURE_OPENAI_API_KEY = request.form["AZURE_OPENAI_API_KEY"]
    PROXYCURL_API_KEY = request.form["PROXYCURL_API_KEY"]
    SERPAPI_API_KEY = request.form["SERPAPI_API_KEY"]
    azure_deployment= request.form["azure_deployment"]
    api_version = request.form["api_version"]
    azure_endpoint = request.form["azure_endpoint"]

    os.environ["AZURE_OPENAI_API_KEY"]=AZURE_OPENAI_API_KEY
    os.environ["PROXYCURL_API_KEY"]=PROXYCURL_API_KEY
    os.environ["SERPAPI_API_KEY"]=SERPAPI_API_KEY
    os.environ["azure_deployment"]= azure_deployment 
    os.environ["api_version"]=api_version
    os.environ["azure_endpoint"]=azure_endpoint
    

    name = request.form["name"]
    summary_and_facts, interests, ice_breakers, profile_pic_url = ice_break(
        name=name
    )
    return jsonify(
        {
            "summary_and_facts": summary_and_facts,
            "interests": interests,
            "ice_breakers": ice_breakers,
            "picture_url": profile_pic_url,
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
