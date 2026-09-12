from flask import Flask
from config import Config

app = Flask(__name__)
app.config.from_object(Config)

from app import views
from app.preload import preload_command

app.cli.add_command(preload_command)