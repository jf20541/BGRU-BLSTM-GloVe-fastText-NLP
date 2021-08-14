from flask import Flask, jsonify, request, render_template
import torch
import numpy as np
import config


app = Flask(__name__)

model = torch.load(config.MODEL_PATH)
model.eval()
