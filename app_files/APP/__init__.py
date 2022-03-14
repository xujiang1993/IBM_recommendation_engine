from flask import Flask

app = Flask(__name__)
from model import content_based_recommendation_engine
from APP import run