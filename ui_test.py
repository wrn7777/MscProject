import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import Image, ImageTk

import os
import glob
import json
import pandas as pd
import csv
import torch
from torch.autograd import Variable
from torch.nn import functional as F

from opts import parse_opts_online
from model import generate_model
from mean import get_mean, get_std
from spatial_transforms import *
from temporal_transforms import *
from target_transforms import ClassLabel
from dataset import get_online_data
from utils import  AverageMeter, LevenshteinDistance, Queue
from types import SimpleNamespace


import pdb
import time
import numpy as np
import datetime

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.btn_import = tk.Button(self)
        self.btn_import["text"] = "Import video"
        self.btn_import.grid(column=0, row=0)

        self.file_result = tk.Label(self,  width=40)
        self.file_result["text"] = "File path:"
        self.file_result.grid(column=1, row=0)


        self.btn_classify = tk.Button(self)
        self.btn_classify["text"] = "Classify Hand Gesture"
        self.btn_classify.grid(column=0, row=1)
        self.img = ImageTk.PhotoImage(Image.open('./res/1.png'))
        self.label_result = tk.Label(self, width=200, height=200, image=self.img)
        self.label_result["text"] = "The recognition result is :"
        self.label_result.grid(column=1, row=1)
        

        self.quit = tk.Button(self, text="QUIT", fg="red",
                              command=self.master.destroy)
        self.quit.grid(column=0, row=2)

        self.time_result = tk.Label(self,  width=40)
        self.time_result["text"] = "Time spend :"
        self.time_result.grid(column=1, row=2)



    

    


root = tk.Tk()
app = Application(master=root)
app.mainloop()