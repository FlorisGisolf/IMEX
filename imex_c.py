# -*- coding: utf-8 -*-
"""
Created on Wed May 16 13:20:32 2018

@author: ovv
"""
import time
import shutil
from tkinter import *
from PIL import Image, ImageTk, ExifTags, ImageFile
import pandas as pd
import numpy as np
#from tkintertable import TableCanvas
import math
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib as mpl
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg#, NavigationToolbar2TkAgg
#import networkx as nx
import pickle
from tkinter import ttk, filedialog
import os
import io
import string
import random
import glob
import torch
import copy
import torchvision.models as models
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.optim as optim

import bottleneck as bn
import webbrowser
from PIL.ExifTags import TAGS, GPSTAGS
from itertools import chain
import torch.utils.data as data
ImageFile.LOAD_TRUNCATED_IMAGES = True
from sklearn.neighbors import typedefs
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import sklearn.utils._cython_blas
import sklearn.neighbors.quad_tree
import sklearn.tree
import sklearn.tree._utils
#from sklearn.decomposition import PCA
#for graphs
#import matplotlib
from collections import defaultdict
import seaborn as sns

batch_size  = 8
num_workers = 4
num_c = 0
tuu=-1
#image_= []
def numpy_unique_ordered(seq):
            array_unique = np.unique(seq, return_index=True)
            dstack = np.dstack(array_unique)
            dstack.dtype = np.dtype([('v', dstack.dtype), ('i', dstack.dtype)])
            dstack.sort(order='i', axis=1)
            return dstack.flatten()['v'].tolist()

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))
    #id generator for exporting images with the same name.

#Main app window
class Application(Frame,object):
    # Define settings upon initialization. Here you can specify
    def __init__(self, master):
        self.themainframe = super(Application,self).__init__(master)
        #some initial parameters
        self.refresh = 0            #keeps track of whether the graph has been updated recently or not
        self.im_list = []           #list with images
        self.df = []                #clustering results
        self.lijstje = []           #list with buckets
        self.theBuckets = []        #the buckets containing the images
        self.bucketDisp = 0         # is 1 when a bucket is displayed, 0 otherwise.
        self.allimages = []         #if images are Preloaded, this contains the images.
        self.num_clus = []          #the number of the current cluster
        self.cm = []                #the array with correlations between images
        self.features= []           #the features of all the images extracted with the neural network
        self.catList = []           #similar to lijstje
        self.loaded_imgs = []       #contains all the actual image data, if preloaded
        self.imagex = []            #contains the images currently displayed
        self.preloaded = 0          #keeps track of whether images are preloaded or not
        self.meta_load = 0          #keeps track of whether the metatab has been loaded or not
        self.rectangles = []        #contains the squares to display currently selected images
        self.greentangles = []      #contains the squares to display current images that are already in a bucket
        self.selected_images = []   #list of selected images
        self.grid()
        self.master.title("imEx")   #title of the app
#        self.image_distance = 10    #distance in pixels between images displayed
        self.focusclick = 0         #keeps track of creating the square on a focused image
        self.rectanglewidth = 3     #width of the red selection rectangles
        self.neuralnet = 'resnet152'#determines the neural network used.
        self.subcyes = 0            # is 1 when subclustering, 0 otherwise
        self.tsneclick = 0          #keeps track of creating the square on tsne graph
        self.X_embed = []         # variable for TSNE embedding
        self.wt = 0
        self.selectedcat = []       #needed to make it possible to deselect item in category box
        self.current_event = 0      #for relevance feedback.
        #currently available:
            #'resnet18'
            #'resnet152' << prefered option
            #'vgg16'
            #'vgg19'
            #'inceptionV-3'
            #'squeezenet'
            #'alexnet'
            #'densenet161' << UNTESTED
        
        style = ttk.Style()
        style.theme_use('default')
        style.configure('.',background = '#555555',foreground = 'white') #sets background of the app to a dark theme
        style.map('.',background=[('disabled','#555555'),('active','#777777')], relief=[('!pressed','sunken'),('pressed','raised')])
        style.configure('TNotebook.Tab',background='#555555')
        style.map('TNotebook.Tab',background=[('selected','black')])
#

#        style.settings()
#        tabstyle = ttk.Style()
#        mygreen = "#d2ffd2"

#
#        tabstyle.theme_create( "yummy", parent="alt", settings={
#            "TNotebook": {"configure": {"tabmargins": [2, 5, 2, 0] } },
#            "TNotebook.Tab": {
#                "configure": {"padding": [5, 1], "background": mygreen },
#                "map":       {"background": [("selected", myred)],
#                              "expand": [("selected", [1, 1, 1, 0])] } } } )
#
#        tabstyle.theme_use("yummy")

        
        self.create_widget()

    def create_widget(self):
        
#        self.notebook = ttk.Notebook()
#         = ttk.Frame(self.notebook)
#        .configure(width=10000, height=10000)        
#        self.notebook.add(text='Cluster viewer')
#        self.notebook.place(x=0,y=1)
#        self.tab2 = ttk.Frame(self.notebook)
#        self.notebook.add(self.tab2,text='Metadata viewer')
        #########################################
        ######## ALL tab1 WIDGETS ###############
        #########################################        
        self.image_distance = 10

        # text displaying some useful tips and a welcome message
        self.communication_label = Message()
        self.communication_label['background'] = '#FFFFFF'        
        self.communication_label['font'] = 24
        self.communication_label.place(x=1150,y=20)
        self.communication_label['width'] = 300
        self.communication_label.configure(text='Hello and welcome to imEx. Start on the left by selecting a folder with images or by loading a previous session. By right clicking a button, you can see some additional information about what the button does.')
        
        #button explanation text. The user can right-click a button to get this explanation
        def open_folder_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='With this button you can select the image folder containing the images you want to analyze. imEx will look through the selected folder and all its subfolders for files with the following extentions: .jpg, .png, .bmp, .gif, .tif')
        def features_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='With this button imEx will extract the features of the images it found in the selected folder using a neural network. The features are a way to represent the content of an image.')
        def cluster_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='The previously calculated features will be used to calculated the similarity between images and generate clusters. Images above the threshold will be placed in the same cluster. If the clusters are too large, and contain a lot of false positives, increasing the threshold may help. If, on the other hand, there are too many, and often small clusters, you can try decreasing the threshold.')
        def load_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='Load a previously saved session and continue where you stopped last time.')
        def addcategory_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='Add a category bucket, so that you can structure your image collection. For example, in a marine safety investigation, you could create a category bucket named \'bridge photos\' and place all photos from the bridge in it. Another useful category bucket could be \'documents\'. You can add images to the buckets using the \'Add cluster to selected bucket(s)\' button, or the \'Add selection to selected bucket(s)\' button.')
        def showbucket_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='Show the content of the selected bucket(s). This also allows you to delete images from the buckets and in a future update, use the bucket to find more relevant instances.')
        def export_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='Export the content of your buckets to a location of your choosing. A folder will be created for each bucket, and a copy of the original images will be placed in the folders.')
        def showimg_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='This button shows the content of the current cluster. You can set how many images you see in the entry box above this button.')
        def addcluster_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='With this button you can add your current cluster to one or more category buckets that you have selected from the list in the middle of the screen. This adds the entire cluster to the bucket at once. If you want more control, you can make a selection, and use the \'Add selection to selected bucket(s)\' button.')
        def addselection_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='With this button you can add your selected images to one or more category buckets that you have selected from the list in the middle of the screen.')
        def delete_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='With this button you can delete selected images from a bucket. To do this you first need to use the \'Show bucket\' button, and it only works if you are viewing a single bucket.')
        def numim_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='Here you can set how many images you want to be shown at the same time. The default is 20. Set to zero if you want to view all images in a cluster or bucket. Note that it can take a moment to load all images when the number is high.')
        def vector_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='Press this button to calculate a representative image for each cluster. This allows you to use the \'Show overview\' button, which shows a single representative image for each cluster. Here you can more quickly see which cluster might be of interest. Note that you only need to use this button once for the calculation, but it might take a moment to calculate.')
        def overview_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='Press this button to show the overview after you have calculated the overview. It displays a representative image for each cluster. Here you can more quickly see which cluster might be of interest. After you have selected a cluster of interest, press the \'Show selected cluster\' button to view the cluster. For now, only a single cluster can be viewed at a time.')
        def selectcluster_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='Press this button to view the cluster you selected from the overview. You can always return to the overview by pressing the \'Show overview\' button again.')
        def focus_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='If only a part of the image is of particular interest, you can click this button to enlarge the image. Next, you can draw a square around the part, and press the \'Query selection\' button. All images in your collection will be ranked on similarity to the part you selected. E.g., if you have an image of a mountain view with a car in the distance, and you are actually interested in the car, you can draw a square around the car and query it to try and find more cars in your image collection.')
        def query_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='All images in your collection will be ranked on similarity to the part you selected after pressing the \'Focus image\' button.')
        def rank_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='All images in your collection will be ranked on similarity to the selected image.')
        def save_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='This button will save your current session. The following things are saved: the list of images from the folder you selected, the buckets you created and the images you added to the buckets, and the cluster you are currently at.')
        def preload_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='This button will load thumbnails of all the images in the folder you selected. This will greatly increase the speed of displaying images, and will result in a more pleasurable experience. However, preloading all images may take some time initially.')
        def nonrelevant_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='If you have added images to the \'NonRelevant\' bucket, you can check this box to make sure the images are not displayed anymore, for example when viewing clusters, or when ranking images.')
        def inbucket_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='Check this box to hide all images that you already placed in a bucket. This is useful for example if you are are ranking images, and only want to find new instances of images not yet in a bucket.')
        def filter_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='By selecting one or more buckets from the left list, only images present in all of the selected buckets will be shown. Additionally, by selecting a bucket on the right, images in those buckets will not be shown. The lists function as AND (left) and NOT (right).')
        def sankey_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='With this button a sankey graph can be created. For each bucket, it will show which fraction of the images is also present in other buckets.')

            
            


        #button to select image folder
        self.open_ = ttk.Button()
        self.open_['text'] ="Select image folder"
        self.open_['command'] = self.select_foldere
        self.open_.bind('<Button-3>',open_folder_text)
        self.open_.place(x=0,y=70)
        self.open_['width'] = 30
        
        # text displaying selected image folder
        self.sel_folder = StringVar()
        self.selected_folder_label = ttk.Label()
        self.selected_folder_label['textvariable'] = self.sel_folder
        self.selected_folder_label['width'] = 60
        self.selected_folder_label.place(x=0,y=50)
        self.sel_folder.set('Select a folder')

        #button to calculate features 
        self.features_ = ttk.Button()
        self.features_['text'] ="Calculate image features"
        self.features_['command'] = self.calculate_features
        self.features_.bind('<Button-3>',features_text)
        self.features_.place(x=0, y=120)
        self.features_['width'] = 30
                          
        #button to start clustering
        self.clustering = ttk.Button()
        self.clustering['text'] ="Cluster images"
        self.clustering['command'] = lambda:[self.cluster_images(),self.showImg()]
        self.clustering.bind('<Button-3>',cluster_text)
        self.clustering.place(x=0,y=170)
        self.clustering['width'] = 30
        
        #button to load
        self.lb = ttk.Button()
        self.lb['text'] ="Load session"
        self.lb['command'] = lambda:[self.load_as(),self.showImg()]
        self.lb.bind('<Button-3>',load_text)
        self.lb.place(x=200,y=70)
        self.lb['width'] = 30

        #button to add new category bucket
        self.bz = ttk.Button()
        self.bz['text'] ="Add bucket >"
        self.bz['command'] = self.addCategory
        self.bz.bind('<Button-3>',addcategory_text)
        self.bz.place(x=450,y=70)
        self.bz['width'] = 19

        #enter name of a new category bucket
        self.e2 = Entry(background='#777777',foreground = 'white',exportselection=0)
        self.e2.insert(END, 'category name')
        self.e2.place(x=450,y=50)

        #button to show currently selected bucket
        self.b4 = ttk.Button()
        self.b4['text'] ="Show bucket"
        self.b4.bind('<Button-3>',showbucket_text)        
        self.b4['command'] = self.showBucket
        self.b4.place(x=450,y=120)
        self.b4['width'] = 19

        #button to export buckets
        self.export = ttk.Button()
        self.export['text'] ="Export buckets"
        self.export['command'] = self.export_buckets
        self.export.bind('<Button-3>',export_text)        
        self.export.place(x=450,y=170)
        self.export['width'] = 19

        # button to go to next cluster
        self.b1 = ttk.Button(command = lambda:[self.nextCluster(),self.showImg()])
        self.b1['text']="Next cluster >"
        self.b1.place(x=400,y=270)
        self.b1['width'] = 30
        #self.bind_all('<Right>',lambda event:[self.nextCluster(),self.showImg()])
        # button to go to previous cluster
        self.bb = ttk.Button(command = lambda:[self.prevCluster(),self.showImg()])
        self.bb['text']="< Previous cluster"
        self.bb.place(x=0,y=270)
        self.bb['width'] = 30
        #self.bind_all('<Left>',lambda event:[self.prevCluster(),self.showImg()])
               
        self.bx = ttk.Button()
        self.bx['text'] ="Show images"  
        self.bx.bind('<Button-3>',showimg_text)        
        self.bx['command'] = self.showImg
        self.bx.place(x=200,y=270)
        self.bx['width'] = 30

        self.bshow = ttk.Button(command = lambda:[self.showCluster(),self.showImg()])
        self.bshow['text'] ="Show cluster"  
        self.bshow.bind('<Button-3>',showimg_text)        
        self.bshow.place(x=970,y=230)
        self.bshow['width'] = 15
        


        
        #button to add current cluster to category bucket        
        self.currentcluster = []
        self.b3 = ttk.Button(command = lambda:[self.addCluster2bucket()])
        self.b3['text'] ="Add cluster to selected bucket(s)"
        self.b3.bind('<Button-3>',addcluster_text)        
        self.b3.place(x=700,y=270)
        self.b3['width'] = 33

        #button to add current selection of images to category bucket        
        self.b_t = ttk.Button(command = lambda:[self.addSelection2bucket()])
        self.b_t.bind('<Button-3>',addselection_text)        
        self.b_t['text'] ="Add selection to selected bucket(s)"
        self.b_t.place(x=920,y=270)
        self.b_t['width'] = 33

        #button to delete selected items from bucket
        self.bDel = ttk.Button()
        self.bDel['text'] ="Delete selected image(s) from bucket"
        self.bDel.bind('<Button-3>',delete_text)        
        self.bDel['command'] = self.click_del
        self.bDel.place(x=1140,y=270)
        self.bDel['width'] = 33
        
        self.ecluster = Entry(background='#777777',foreground = 'white',exportselection=0)
        self.ecluster.insert(END, 0) #0 is default, change if needed. 
        self.ecluster.bind('<Button-3>',numim_text)        
        self.ecluster.place(x=900,y=230)    
        self.ecluster['width'] = 10

        #enter number of images shown
        self.e1 = Entry(background='#777777',foreground = 'white',exportselection=0)
        self.e1.insert(END, 20) #20 is default, change if needed. 
        self.e1.bind('<Button-3>',numim_text)        
        self.e1.place(x=200,y=250)
        self.num_clus = 0
        
        self.var = IntVar()
        self.var.set(0)
        
        #button to rank image
        self.rank_im_button = ttk.Button()
        self.rank_im_button['text'] ="Rank selected image"
        self.rank_im_button.bind('<Button-3>',rank_text)        
        self.rank_im_button['command'] = self.rank_image
        self.rank_im_button.place(x=900,y=20)
        self.rank_im_button['width'] = 30

        #button to subcluster images
        self.subc = ttk.Button()
        self.subc['text'] ="Subcluster"
        self.subc.bind('<Button-3>',rank_text)        
        self.subc['command'] = self.subcluster
        self.subc.place(x=800,y=20)
        self.subc['width'] = 15

        #button to super rank images
        self.subc = ttk.Button()
        self.subc['text'] ="Super rank"
        self.subc.bind('<Button-3>',rank_text)        
        self.subc['command'] = self.rank_bucket
        self.subc.place(x=800,y=50)
        self.subc['width'] = 15



        #button to calculate cluster overview
        self.bO = ttk.Button()
        self.bO['text'] ="Calculate overview"
        self.bO.bind('<Button-3>',vector_text)        
        self.bO['command'] = self.calculate_avg_vector
        self.bO.place(x=900,y=50)
        self.bO['width'] = 30

        #button to show cluster overview
        self.bO2 = ttk.Button()
        self.bO2['text'] ="Show overview"
        self.bO2.bind('<Button-3>',overview_text)        
        self.bO2['command'] = self.show_overview
        self.bO2.place(x=900,y=80)
        self.bO2['width'] = 30
                 
        #button to show cluster selected from the overview
        self.bO3 = ttk.Button()
        self.bO3['text'] ="Show selected cluster"
        self.bO3.bind('<Button-3>',selectcluster_text)        
        self.bO3['command'] = self.show_selected_cluster
        self.bO3.place(x=900,y=110)
        self.bO3['width'] = 30

        #button to focus on an image, in the focus window you can select part of an image to extact features from
        self.maskim = ttk.Button()
        self.maskim['text'] ="Focus image"
        self.maskim.bind('<Button-3>',focus_text)        
        self.maskim['command'] = self.focus_image
        self.maskim.place(x=900,y=140)
        self.maskim['width'] = 30

        #button to focus on an image
        self.qrmask = ttk.Button()
        self.qrmask['text'] ="Query selection"
        self.qrmask.bind('<Button-3>',query_text)        
        self.qrmask['command'] = self.query_selection
        self.qrmask.place(x=900,y=170)
        self.qrmask['width'] = 30


        #button to focus on an image
        self.external = ttk.Button()
        self.external['text'] ="Query external image"
        self.external.bind('<Button-3>',query_text)        
        self.external['command'] = self.query_external
        self.external.place(x=900,y=200)
        self.external['width'] = 30


        #button to save
        self.sb = ttk.Button()
        self.sb['text'] ="Save session as"
        self.sb.bind('<Button-3>',save_text)        
        self.sb['command'] = self.save_as
        self.sb.place(x=200,y=120)
        self.sb['width'] = 30

#        #button to preload all images
        self.preloadbutton = ttk.Button()
        self.preloadbutton['text'] ="Preload images"
        self.preloadbutton['command'] = self.preload2
        self.preloadbutton.bind('<Button-3>',preload_text)        
        self.preloadbutton.place(x=200,y=170)
        self.preloadbutton['width'] = 30

        #button to expand current cluster
        self.expand_c = ttk.Button()
        self.expand_c['text'] ="Expand current cluster"
        self.expand_c['command'] = self.expand_cluster
        self.expand_c.bind('<Button-3>',preload_text)        
        self.expand_c.place(x=200,y=200)
        self.expand_c['width'] = 30

               
        #button to show or hide images in the Non-relevant bucket
        vv = IntVar()
        self.nonR = ttk.Checkbutton(variable = vv)
        self.nonR['text'] ="Check to hide Non-relevant images from displayed results"
        self.nonR.bind('<Button-3>',nonrelevant_text)        
        self.nonR.var = vv
        self.nonR.place(x=400,y=220)
        
        #button to show or hide images already in a bucket
        ww = IntVar()
        self.inbucket = ttk.Checkbutton(variable = ww)
        self.inbucket['text'] ="Check to hide images already in a bucket"
        self.inbucket.bind('<Button-3>',inbucket_text)        
        self.inbucket.var = ww
        self.inbucket.place(x=400,y=240)


        ### some other stuff ####
        #creates window for statistics and other data
        self.newWindow = Toplevel(self.master)
        self.newWindow.geometry("1400x500")
        self.newWindow.title("Buckets")
        self.newWindow.configure(background='#555555')
        #create dataframe for the bucket
        self.theBuckets = {}
        #list of available category buckets
        valores = StringVar()
        valores.set("RelevantItems Non-RelevantItems")
        self.theBuckets["RelevantItems"] = []
        self.theBuckets["Non-RelevantItems"] = []
        #scrollbar for the buckets
        boxscrollbar = Scrollbar(width = 10)
        #listbox containing the names of the buckets
        self.categories = Listbox(width=30,background='#777777',foreground='white',yscrollcommand=boxscrollbar.set,exportselection=0)
        self.categories['listvariable'] = valores
        self.categories['selectmode'] = 'extended'               
        self.categories.place(x=585,y=50)    
        #place of the scrollbar
        boxscrollbar.config(command=self.categories.yview)
        boxscrollbar.place(in_=self.categories,relx=1.0, relheight=1)
        self.categories.bind('<Button-1>', self.deselect_list )
        
        #filter the self.categories listbox
        self.search_var = StringVar()
        self.search_var.trace_add("write", self.update_the_list)
        self.filter_entry = Entry(background='#777777',foreground = 'white', textvariable=self.search_var, width=30)
        self.filter_entry.place(x=585,y=30)
        self.filter_cats_label = ttk.Label(width=30)
        self.filter_cats_label['text'] = 'Filter the buckets'
        self.filter_cats_label.place(x=585,y=15)
        
#        #entryform to change name of a bucket
#        self.changename = Entry(background='#777777',foreground = 'white', width=30)
#        self.changename.place(x=0, y=0)
        #button to change name of a bucket
        self.changebutton = ttk.Button()                                    
        self.changebutton['text'] = "Change bucket name"
        self.changebutton['command'] = self.change_name
        self.changebutton.place(x=450, y=20)
        
        
        #set the threshold for clustering
        self.threshold = 0.5 #0.5 is a good default. Increase for more, but smaller clusters, usually with higher precision. Decrease for fewer and larger clusters, usually with lower precision
        self.threshold_entry = Entry(background='#777777',foreground = 'white',exportselection=0)
        self.threshold_entry.insert(END, 0.5)
        self.threshold_entry.place(x=0,y=220)

        #label for the threshold
        self.set_threshold_label = ttk.Label()
        self.set_threshold_label['text'] = 'Set threshold between 0 and 1'
        self.set_threshold_label.place(x=0,y=200)

        #enter the size of the images displayed. 100 is default
        self.imsize = 100
        self.imsize_entry = Entry(background='#777777',foreground = 'white',exportselection=0)
        self.imsize_entry.insert(END, 100)
        self.imsize_entry.place(x=1360,y=270)
        self.imsize_entry['validate'] = 'focusout'
        self.imsize_entry['validatecommand'] = self.get_imsize     
        
        self.set_imsize_label = ttk.Label()
        self.set_imsize_label['text'] = 'Set the image display\n size in pixels'
        self.set_imsize_label['wraplength'] = 200
        self.set_imsize_label['justify'] = CENTER
        self.set_imsize_label.place(x=1360,y=240)
               
        #canvas for the images, which adjusts to screen width. Optimized for windows. For sidebar, like in linux, you may want to decrease screen width
        self.screen_width = root.winfo_screenwidth() #requests screen width
        self.screen_width = self.screen_width-60
        self.screen_height = root.winfo_screenheight() #requests screen height
        self.screen_height = self.screen_height-400
        yscrollbar = Scrollbar(width = 16) #scroll  bar for canvas
        self.c = Canvas(bg='#666666',bd=0, scrollregion=(0, 0, 0, 500), yscrollcommand=yscrollbar.set, width =self.screen_width, height =self.screen_height) #canvas size
        self.c.place(x = 0, y=300)
        yscrollbar.config(command=self.c.yview)
        yscrollbar.place(in_=self.c,relx=1.0, relheight=1)
        self.num_im_row = math.floor(self.screen_width / (self.imsize + self.image_distance)) #the total number of images that fit from left to right
        
        #binds scrollwheel when scrolling with mouse on canvas
        self.c.bind('<Enter>', self._bound_to_mousewheel)
        self.c.bind('<Leave>', self._unbound_to_mousewheel)

        #
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.origX = self.c.xview()[0]
        self.origY = self.c.yview()[0]

        #progressbar. Doesnt work yet
        self.progress = ttk.Progressbar(self, orient='horizontal', length = 200, mode='determinate')
        self.progress.place(x=10,y=10)
    
        
        # second window stuff
        
        self.progress_label = ttk.Label(self.newWindow)
        self.progress_label['text'] = 'Progress'
        self.progress_label.place(x=125,y=280)
        
        #button to show filtered buckets
        self.filter_button = ttk.Button(self.newWindow)
        self.filter_button['text'] ="Show filtered buckets"
        self.filter_button['command'] = self.filter_buckets
        self.filter_button.bind('<Button-3>',filter_text)        
        self.filter_button.place(x=125,y=220)
        self.filter_button['width'] = 30

        self.sankey_button = ttk.Button(self.newWindow)
        self.sankey_button['text'] ="Create sankey diagram"
        self.sankey_button['command'] = self.create_sankey
        self.sankey_button.bind('<Button-3>',sankey_text)        
        self.sankey_button.place(x=125,y=250)
        self.sankey_button['width'] = 30

        self.tsne_button = ttk.Button(self.newWindow)
        self.tsne_button['text'] ="Create tsne graph"
        self.tsne_button['command'] = self.create_tsne
        self.tsne_button.bind('<Button-3>',sankey_text)        
        self.tsne_button.place(x=400,y=250)
        self.tsne_button['width'] = 30

#        self.graph_button = ttk.Button(self.newWindow)
#        self.graph_button['text'] ="Create graph"
#        #self.graph_button['command'] = self.creategraph
#        self.graph_button.bind('<Button-3>',sankey_text)        
#        self.graph_button.place(x=125,y=280)
#        self.graph_button['width'] = 30

        
        self.filter_in_label = ttk.Label(self.newWindow)
        self.filter_in_label['text'] = 'Show images only in:'
        self.filter_in_label.place(x=20,y=20)

        self.filter_out_label = ttk.Label(self.newWindow)
        self.filter_out_label['text'] = 'Show images NOT in:'
        self.filter_out_label.place(x=250,y=20)

        
        
        boxscrollbar2 = Scrollbar(self.newWindow,width = 10)
        #listbox containing the names of the buckets
        self.categories2 = Listbox(self.newWindow,width=30,background='#777777',foreground='white',yscrollcommand=boxscrollbar2.set,exportselection=1)
        self.catList = self.categories.get(0,END)
        self.categories2['listvariable'] = self.catList
        for k in range(0,len(self.catList)):
            self.categories2.insert(END,self.catList[k])
        self.categories2['selectmode'] = 'extended'               
        self.categories2.place(x=20,y=50)    
        #place of the scrollbar
        boxscrollbar2.config(command=self.categories2.yview)
        boxscrollbar2.place(in_=self.categories2,relx=1.0, relheight=1)
#
        boxscrollbar3 = Scrollbar(self.newWindow,width = 10)
        #listbox containing the names of the buckets
        self.categories3 = Listbox(self.newWindow,width=30,background='#777777',foreground='white',yscrollcommand=boxscrollbar3.set,exportselection=0)
        self.catList = self.categories.get(0,END)
        self.categories3['listvariable'] = self.catList
        for k in range(0,len(self.catList)):
            self.categories3.insert(END,self.catList[k])
        self.categories3['selectmode'] = 'extended'               
        self.categories3.place(x=250,y=50)    
        #place of the scrollbar
        boxscrollbar3.config(command=self.categories3.yview)
        boxscrollbar3.place(in_=self.categories3,relx=1.0, relheight=1)
    

    
    
        
        #this def updates the categories on the second window, as well as the graph.
        def update_cats(event):
            self.catList = self.categories.get(0,END)
            self.categories2.delete(0,'end')
            self.categories3.delete(0,'end')
            for k in range(0,len(self.catList)):
                self.categories2.insert(END,self.catList[k])
                #self.categories3.insert(END,self.catList[k])
            
            if self.refresh == 0:
                self.refresh = 1
                self.f = Figure( figsize=(self.screen_width/100/2-1.5,self.screen_height/100), dpi=100,facecolor='#555555' )
                self.a = self.f.add_subplot(1,1,1, facecolor='#555555')
                bar_y = list(self.theBuckets.keys())
                bar_x = []
                for u in range (0,len(bar_y)):
                    try:
                        bar_x.append(len(self.theBuckets[bar_y[u]]))
                    except TypeError:
                        bar_x.append(1)
                y_pos = np.arange(len(bar_y))
                for i, v in enumerate(bar_x):
                    self.a.text(3, i + .25, str(v), color='#FFFFFF')
                self.a.barh(y_pos,bar_x)
                self.a.set_yticks(y_pos)
                self.a.set_yticklabels(bar_y,color='#FFFFFF')
                self.a.invert_yaxis()
                self.a.set_xlabel('Number of images in bucket',color='#FFFFFF')
                self.a.tick_params(axis='x', colors='#FFFFFF')
                self.a.tick_params(axis='y', colors='#FFFFFF')
                self.a.spines['bottom'].set_color('#FFFFFF')
                self.a.spines['top'].set_color('#FFFFFF') 
                self.a.spines['right'].set_color('#FFFFFF')
                self.a.spines['left'].set_color('#FFFFFF')
                self.a.grid(b=None)
                self.canvasG = Canvas(self.newWindow,bg='#666666',bd=0, width = self.screen_width/2, height =self.screen_height/2-200) #canvas size
                self.canvasG.place(x = 0, y=320)
                self.f.tight_layout()
                self.gcanvas = FigureCanvasTkAgg(self.f, master=self.canvasG)
                #self.gcanvas.show()
                self.gcanvas.get_tk_widget().pack(side=BOTTOM, fill=BOTH, expand=1)
                
                unique_values = sorted(list(chain.from_iterable(self.theBuckets.values())))
                
                if len(self.im_list) > 0:
                    self.progress_label.configure(text='You have placed '+ str(len(unique_values)) + ' out of ' + str(len(self.im_list)) + ' images in buckets' + ' (' + str(round(len(unique_values)/len(self.im_list)*100,1)) +'%).')
            
        self.newWindow.bind('<FocusIn>', update_cats)
        
        
        def in_filter(event):
            self.filter_in = self.categories2.curselection()
        self.categories2.bind('<<ListboxSelect>>', in_filter)

        def out_filter(event):
            self.filter_out = self.categories3.curselection()
        self.categories3.bind('<<ListboxSelect>>', out_filter)
        
        def creategraph(event): #still under construction        
            self.f = Figure( figsize=(self.screen_width/100/2-1.5,self.screen_height/100), dpi=100,facecolor='#555555' )
            self.a = self.f.add_subplot(1,1,1, facecolor='#555555')
            bar_y = list(self.theBuckets.keys())
            bar_x = []
            for u in range (0,len(bar_y)):
                try:
                    bar_x.append(len(self.theBuckets[bar_y[u]]))
                except TypeError:
                    bar_x.append(1)
            y_pos = np.arange(len(bar_y))
            for i, v in enumerate(bar_x):
                self.a.text(3, i + .25, str(v), color='#FFFFFF')
    
            self.a.barh(y_pos,bar_x)
            self.a.set_yticks(y_pos)
            self.a.set_yticklabels(bar_y,color='#FFFFFF')
            self.a.invert_yaxis()
            self.a.set_xlabel('Number of images in bucket',color='#FFFFFF')
            self.a.tick_params(axis='x', colors='#FFFFFF')
            self.a.tick_params(axis='y', colors='#FFFFFF')
            self.a.spines['bottom'].set_color('#FFFFFF')
            self.a.spines['top'].set_color('#FFFFFF') 
            self.a.spines['right'].set_color('#FFFFFF')
            self.a.spines['left'].set_color('#FFFFFF')
            self.canvasG = Canvas(self.newWindow,bg='#666666',bd=0, width = self.screen_width/2, height =self.screen_height/2-200) #canvas size
            self.canvasG.place(x = 0, y=320)
            #self.f.tight_layout()
            self.gcanvas = FigureCanvasTkAgg(self.f, master=self.canvasG)
    #        self.gcanvas.show()
            self.gcanvas.get_tk_widget().pack(side=BOTTOM, fill=BOTH, expand=1)
        
#        self.newWindow.bind('<FocusIn>', creategraph)

                        
        def open_image3(event):
            evex = self.c.canvasx(event.x)
            evey = self.c.canvasy(event.y)
            x_num = math.ceil((evex)/(self.imsize + self.image_distance))-1
            y_num = math.ceil((evey)/(self.imsize + self.image_distance))
            im_num = x_num + self.num_im_row*(y_num-1)
            im_tag = self.c.gettags(self.imagex[im_num])
            im_tag = int(float(im_tag[0]))
            mijn_plaatje = self.im_list[im_tag]
            webbrowser.open(mijn_plaatje)
#
        self.c.bind('<Double-Button-1>', open_image3)

        #function to select an image and display a red square around the selected image
        def click_select(event):
            if self.bucketDisp == 5: #in focusview (bucketDisp=5), this function allows you to draw a square
                self.focusclick = self.focusclick + 1
                if self.focusclick == 2:
                    self.evex2 = self.c.canvasx(event.x)
                    self.evey2 = self.c.canvasy(event.y)
                    self.focusclick = 0
                    if self.squares is not None:
                        self.c.delete(self.squares)
                    self.squares = self.c.create_rectangle(self.evex1, self.evey1, self.evex2, self.evey2)                    
                else:
                    self.evex1 = self.c.canvasx(event.x)
                    self.evey1 = self.c.canvasy(event.y)
            else: #if not in focusview, this function allows you to select images and draws a red square around the selected image
                for q in range(0,len(self.rectangles)):
                    self.c.delete(self.rectangles[q])
                evex = self.c.canvasx(event.x)
                evey = self.c.canvasy(event.y)
                x_num = math.ceil((evex)/(self.imsize + self.image_distance))-1
                y_num = math.ceil((evey)/(self.imsize + self.image_distance))
                self.im_numX = x_num + self.num_im_row*(y_num-1) 
                self.selected_images = []
                self.rectangles = []
                self.selected_images.append(self.im_numX)
                row_ = math.floor(self.im_numX/self.num_im_row)
                column_ = self.im_numX%self.num_im_row
                if len(self.imagex) >= self.im_numX+1:
                    self.rectangles.append(self.c.create_rectangle(column_*(self.imsize + self.image_distance), row_*(self.imsize + self.image_distance), column_*(self.imsize + self.image_distance)+(self.imsize + self.image_distance)-self.image_distance, row_*(self.imsize + self.image_distance)+(self.imsize + self.image_distance)-self.image_distance, outline='red',width=self.rectanglewidth,tags = self.im_numX))                
        self.c.bind("<Button-1>", click_select)
        
        #this allows you to select multiple adjacent images using the shift key + mouse button 1
        def shift_click_select(event):
            evex = self.c.canvasx(event.x)
            evey = self.c.canvasy(event.y)
            x_num = math.ceil((evex)/(self.imsize + self.image_distance))-1
            y_num = math.ceil((evey)/(self.imsize + self.image_distance))
            self.im_numY = x_num + self.num_im_row*(y_num-1)

            for q in range(0,len(self.rectangles)):
                self.c.delete(self.rectangles[q])
            self.rectangles = []
            self.selected_images = []
            im_selected = np.sort(np.array([self.im_numY,self.im_numX]))
            for p in range(im_selected[0],im_selected[1]+1):
                row_ = math.floor(p/self.num_im_row)
                column_ = p%self.num_im_row
                self.selected_images.append(p)
                self.rectangles.append(self.c.create_rectangle(column_*(self.imsize + self.image_distance), row_*(self.imsize + self.image_distance), column_*(self.imsize + self.image_distance)+(self.imsize + self.image_distance)-self.image_distance, row_*(self.imsize + self.image_distance)+(self.imsize + self.image_distance)-self.image_distance, outline='red',width=self.rectanglewidth,tags = p))
            
        self.c.bind("<Shift-Button-1>", shift_click_select)
        
        #this function allows you to select multiple non-adjacent images by hold down control + left click
        def ctrl_click_select(event):
            evex = self.c.canvasx(event.x)
            evey = self.c.canvasy(event.y)
            x_num = math.ceil((evex)/(self.imsize + self.image_distance))-1
            y_num = math.ceil((evey)/(self.imsize + self.image_distance))
            self.im_numX = x_num + self.num_im_row*(y_num-1)
            row_ = math.floor(self.im_numX/self.num_im_row)
            column_ = self.im_numX%self.num_im_row
            current_tags = []
            if self.im_numX in self.selected_images: 
                deselect = self.selected_images.index(self.im_numX)
                self.c.delete(self.rectangles[deselect])
                #mask = np.ones(len(self.selected_images),dtype=bool)
                self.selected_images.remove(self.im_numX)
                #self.selected_images = self.selected_images[mask]
            else:
                self.rectangles.append(self.c.create_rectangle(column_*(self.imsize + self.image_distance), row_*(self.imsize + self.image_distance), column_*(self.imsize + self.image_distance)+(self.imsize + self.image_distance)-self.image_distance, row_*(self.imsize + self.image_distance)+(self.imsize + self.image_distance)-self.image_distance, outline='red',width=self.rectanglewidth,tags = self.im_numX))
                self.selected_images.append(self.im_numX)
                
        self.c.bind("<Control-Button-1>", ctrl_click_select)

#        def tsne_click(event):
#            self.evex_tsne2 = self.canvas_tsne.canvasx(event.x)
#            self.evey_tsne2 = self.canvas_tsne.canvasy(event.y)
#            self.focusclick = 0
#            if self.tsne_squares is not None:
#                self.canvas_tsne.delete(self.tsne_squares)
#                self.tsne_squares = self.canvas_tsne.create_rectangle(self.evex_tsne1, self.evey_tsne1, self.evex_tsne2, self.evey_tsne2)                    
#            else:
#                self.evex_tsne1 = self.canvas_tsne.canvasx(event.x)
#                self.evey_tsne1 = self.canvas_tsne.canvasy(event.y)
#            print('HELLO')
            

        
        #this function ranks a selected image by rightclicking an image. It will sort all images based on correlation
        def rank_images(event):
            self.communication_label.configure(text='Calculating the ranking. Please wait.')
            self.communication_label['background'] = '#99CCFF'
            self.communication_label.update_idletasks()
            evex = self.c.canvasx(event.x)   #x location to determine selected image
            evey = self.c.canvasy(event.y)   #y location to determine selected image
            self.bucketDisp = 2
            x_num = math.ceil((evex)/(self.imsize + self.image_distance))-1 #determine the row
            y_num = math.ceil((evey)/(self.imsize + self.image_distance))   #determine the column
            im_num = x_num + self.num_im_row*(y_num-1) #calculate the actual image number using row and column
            im_tag = self.c.gettags(self.imagex[im_num])    #get the actual image id from imagex (imagex is a list of all currently displayed images)
            im_tag = int(float(im_tag[0]))
            self.rank_list = self.cm[:,int(im_tag)] # get all the correlations between the selected image and all the other images
            temp_list = np.sort(self.rank_list,0)[::-1] #sorts the correlations
            self.rank_list = np.argsort(self.rank_list,0)[::-1] #sorts the id of all the images based on correlation
            self.c.xview_moveto(self.origX)  #####
            self.c.yview_moveto(self.origY) ######
            self.rank_list = np.asarray(self.rank_list)
            temp_list = np.asarray(temp_list)
            temp_list[np.isnan(temp_list)] = -100   #some images have no correlations due to issues with loading and extracting the image features. Ususally means the image file is damaged
            self.rank_list = self.rank_list[temp_list>-50] # this removes all the broken images
            self.rank_list = np.append(im_tag,self.rank_list) # this adds the selected (queried) image to the image list to be displayed
            #self.bucketDisp = 0
            self.c.delete("all")
            self.display_images(self.rank_list) #function to display the ranked list of images. By default it displays all images, but may need to be limited for large datasets

            self.communication_label['background'] = '#FFFFFF'
            self.communication_label.configure(text='Finished calculating. Showing the ranking')
            plt.close("all")

            
        self.c.bind("<Button-3>", rank_images)


    def deselect_list(self, event):
        if len(self.selectedcat) == 0:
            self.selectedcat = self.categories.curselection()
        if len(self.selectedcat) == 1:
            if self.categories.curselection() == self.selectedcat:
                self.categories.selection_clear(0,END)
                self.selectedcat = []        
        
    def _bound_to_mousewheel(self,event):
            self.c.bind_all("<MouseWheel>", self._on_mousewheel)
            
    def _unbound_to_mousewheel(self, event):
            self.c.unbind_all("<MouseWheel>") 

    def update_the_list(self,*args):
        search_term = self.search_var.get()

        self.categories.delete(0, END)
        
        for item in self.catList:
                if search_term.lower() in item.lower():
                    self.categories.insert(END, item)
    
    #the function that is called by other functions in order to display images
    def display_images(self, cluster): #x is the list with the image names, cluster is the list with ids. 
        self.ccluster = cluster


#        self.c.destroy()
#        yscrollbar = Scrollbar(width = 16) #scroll  bar for canvas
#        self.c = Canvas(bg='#666666',bd=0, scrollregion=(0, 0, 0, 500), yscrollcommand=yscrollbar.set, width =self.screen_width, height =self.screen_height) #canvas size
#        self.c.place(x = 0, y=300)
#        yscrollbar.config(command=self.c.yview)
#        yscrollbar.place(in_=self.c,relx=1.0, relheight=1)
#        self.c.bind('<Enter>', self._bound_to_mousewheel)
#        self.c.bind('<Leave>', self._unbound_to_mousewheel)



        def open_image3(event):
            evex = self.c.canvasx(event.x)
            evey = self.c.canvasy(event.y)
            x_num = math.ceil((evex)/(self.imsize + self.image_distance))-1
            y_num = math.ceil((evey)/(self.imsize + self.image_distance))
            im_num = x_num + self.num_im_row*(y_num-1)
            im_tag = self.c.gettags(self.imagex[im_num])
            im_tag = int(float(im_tag[0]))
            mijn_plaatje = self.im_list[im_tag]
            webbrowser.open(mijn_plaatje)
#
        self.c.bind('<Double-Button-1>', open_image3)

        #function to select an image and display a red square around the selected image
        def click_select(event):
            if self.bucketDisp == 5: #in focusview (bucketDisp=5), this function allows you to draw a square
                self.focusclick = self.focusclick + 1
                if self.focusclick == 2:
                    self.evex2 = self.c.canvasx(event.x)
                    self.evey2 = self.c.canvasy(event.y)
                    self.focusclick = 0
                    if self.squares is not None:
                        self.c.delete(self.squares)
                    self.squares = self.c.create_rectangle(self.evex1, self.evey1, self.evex2, self.evey2)                    
                else:
                    self.evex1 = self.c.canvasx(event.x)
                    self.evey1 = self.c.canvasy(event.y)
            else: #if not in focusview, this function allows you to select images and draws a red square around the selected image
                for q in range(0,len(self.rectangles)):
                    self.c.delete(self.rectangles[q])
                evex = self.c.canvasx(event.x)
                evey = self.c.canvasy(event.y)
                x_num = math.ceil((evex)/(self.imsize + self.image_distance))-1
                y_num = math.ceil((evey)/(self.imsize + self.image_distance))
                self.im_numX = x_num + self.num_im_row*(y_num-1) 
                self.selected_images = []
                self.rectangles = []
                self.selected_images.append(self.im_numX)
                row_ = math.floor(self.im_numX/self.num_im_row)
                column_ = self.im_numX%self.num_im_row
                if len(self.imagex) >= self.im_numX+1:
                    self.rectangles.append(self.c.create_rectangle(column_*(self.imsize + self.image_distance), row_*(self.imsize + self.image_distance), column_*(self.imsize + self.image_distance)+(self.imsize + self.image_distance)-self.image_distance, row_*(self.imsize + self.image_distance)+(self.imsize + self.image_distance)-self.image_distance, outline='red',width=self.rectanglewidth,tags = self.im_numX))                
        self.c.bind("<Button-1>", click_select)
        
        #this allows you to select multiple adjacent images using the shift key + mouse button 1
        def shift_click_select(event):
            evex = self.c.canvasx(event.x)
            evey = self.c.canvasy(event.y)
            x_num = math.ceil((evex)/(self.imsize + self.image_distance))-1
            y_num = math.ceil((evey)/(self.imsize + self.image_distance))
            self.im_numY = x_num + self.num_im_row*(y_num-1)

            for q in range(0,len(self.rectangles)):
                self.c.delete(self.rectangles[q])
            self.rectangles = []
            self.selected_images = []
            im_selected = np.sort(np.array([self.im_numY,self.im_numX]))
            for p in range(im_selected[0],im_selected[1]+1):
                row_ = math.floor(p/self.num_im_row)
                column_ = p%self.num_im_row
                self.selected_images.append(p)
                self.rectangles.append(self.c.create_rectangle(column_*(self.imsize + self.image_distance), row_*(self.imsize + self.image_distance), column_*(self.imsize + self.image_distance)+(self.imsize + self.image_distance)-self.image_distance, row_*(self.imsize + self.image_distance)+(self.imsize + self.image_distance)-self.image_distance, outline='red',width=self.rectanglewidth,tags = p))
            
        self.c.bind("<Shift-Button-1>", shift_click_select)
        
        #this function allows you to select multiple non-adjacent images by hold down control + left click
        def ctrl_click_select(event):
            evex = self.c.canvasx(event.x)
            evey = self.c.canvasy(event.y)
            x_num = math.ceil((evex)/(self.imsize + self.image_distance))-1
            y_num = math.ceil((evey)/(self.imsize + self.image_distance))
            self.im_numX = x_num + self.num_im_row*(y_num-1)
            row_ = math.floor(self.im_numX/self.num_im_row)
            column_ = self.im_numX%self.num_im_row
            current_tags = []
            if self.im_numX in self.selected_images: 
                deselect = self.selected_images.index(self.im_numX)
                self.c.delete(self.rectangles[deselect])
                #mask = np.ones(len(self.selected_images),dtype=bool)
                self.selected_images.remove(self.im_numX)
                #self.selected_images = self.selected_images[mask]
            else:
                self.rectangles.append(self.c.create_rectangle(column_*(self.imsize + self.image_distance), row_*(self.imsize + self.image_distance), column_*(self.imsize + self.image_distance)+(self.imsize + self.image_distance)-self.image_distance, row_*(self.imsize + self.image_distance)+(self.imsize + self.image_distance)-self.image_distance, outline='red',width=self.rectanglewidth,tags = self.im_numX))
                self.selected_images.append(self.im_numX)
                
        self.c.bind("<Control-Button-1>", ctrl_click_select)

        #this function ranks a selected image by rightclicking an image. It will sort all images based on correlation
        def rank_images(event):
            self.communication_label.configure(text='Calculating the ranking. Please wait.')
            self.communication_label['background'] = '#99CCFF'
            self.communication_label.update_idletasks()
            evex = self.c.canvasx(event.x)   #x location to determine selected image
            evey = self.c.canvasy(event.y)   #y location to determine selected image
            self.bucketDisp = 2
            x_num = math.ceil((evex)/(self.imsize + self.image_distance))-1 #determine the row
            y_num = math.ceil((evey)/(self.imsize + self.image_distance))   #determine the column
            im_num = x_num + self.num_im_row*(y_num-1) #calculate the actual image number using row and column
            im_tag = self.c.gettags(self.imagex[im_num])    #get the actual image id from imagex (imagex is a list of all currently displayed images)
            im_tag = int(float(im_tag[0]))
            self.rank_list = self.cm[:,int(im_tag)] # get all the correlations between the selected image and all the other images
            temp_list = np.sort(self.rank_list,0)[::-1] #sorts the correlations
            self.rank_list = np.argsort(self.rank_list,0)[::-1] #sorts the id of all the images based on correlation
            self.c.xview_moveto(self.origX)  #####
            self.c.yview_moveto(self.origY) ######
            self.rank_list = np.asarray(self.rank_list)
            temp_list = np.asarray(temp_list)
            temp_list[np.isnan(temp_list)] = -100   #some images have no correlations due to issues with loading and extracting the image features. Ususally means the image file is damaged
            self.rank_list = self.rank_list[temp_list>-50] # this removes all the broken images
            self.rank_list = np.append(im_tag,self.rank_list) # this adds the selected (queried) image to the image list to be displayed
            #self.bucketDisp = 0
            self.c.delete("all")
            self.display_images(self.rank_list) #function to display the ranked list of images. By default it displays all images, but may need to be limited for large datasets

            self.communication_label['background'] = '#FFFFFF'
            self.communication_label.configure(text='Finished calculating. Showing the ranking')
            plt.close("all")

            
        self.c.bind("<Button-3>", rank_images)
        
        self.num_im_row = math.floor(self.screen_width / (self.imsize + self.image_distance)) #the total number of images that fit from left to right

        if self.nonR.var.get() == 1 and self.bucketDisp is not 1:
            nonrelevants = self.theBuckets['Non-RelevantItems']
            self.ccluster = [vg for vg in self.ccluster if vg not in nonrelevants]
            self.ccluster = np.asarray(pd.DataFrame.from_dict(self.ccluster)).squeeze()
        if self.inbucket.var.get() == 1 and self.bucketDisp != 1:
            selected = self.categories.curselection()
            if len(selected) == 0:
                #self.ccluster = [vg for vg in self.ccluster if vg not in list(chain.from_iterable(self.theBuckets.values()))]
                t = list(chain.from_iterable(self.theBuckets.values()))
                xx = list(self.ccluster)
                self.ccluster = list(sorted(set(self.ccluster) - set(t), key=xx.index))
                self.ccluster = np.asarray(self.ccluster)
                
                
#                self.ccluster = np.asarray(pd.DataFrame.from_dict(self.ccluster)).squeeze()
            else:
#                self.ccluster = [vg for vg in self.ccluster if vg not in self.theBuckets[self.categories.get(selected[0])]]
                t = list(chain.from_iterable(self.theBuckets.values()))
                xx = list(self.ccluster)
                self.ccluster = list(sorted(set(self.ccluster) - set(t), key=xx.index))
                self.ccluster = np.asarray(self.ccluster)
#                self.ccluster = np.asarray(pd.DataFrame.from_dict(self.ccluster)).squeeze()
                
#        from sys import getsizeof
#        from collections import OrderedDict, Mapping, Container
#        def deep_getsizeof(o, ids):
#            d = deep_getsizeof
#            if id(o) in ids:
#                return 0
#            r = getsizeof(o)
#            ids.add(id(o))
#            
#            if isinstance(o, str) or isinstance(o, str):
#                return r
#            if isinstance(o, Mapping):
#                return r + sum(d(k,ids) + d(v, ids) for k, v in o.iteritems())
#            if isinstance(o, Container):
#                return r + sum(d(x, ids) for x in o)
#            return r    
        num_im =int(self.e1.get())
        if num_im == 0:
            num_im = len(self.im_list)
        if num_im > 4950:
            num_im = 4950
        if num_im > len(cluster):
            num_im = len(cluster)
        else:
            self.ccluster = self.ccluster[0:num_im]
        x = []
        for ij in range(0,len(self.ccluster)):
            x.append(self.im_list[self.ccluster[ij]])

        self.imagex = []
        self.c['scrollregion'] = (0,0,0,math.ceil(len(x)/self.num_im_row)*(self.imsize+self.image_distance))
        self.c.delete("all")
        self.greentangles = []
        self.purpletangles = []
        if self.bucketDisp == 5:
            self.bucketDisp == 0
        if self.preloaded == 1: #if images are preloaded, the images displayed will be selected from self.allimages

#            try: #### this fixes a memory leak by deleting currently loaded images in the memory.
#                len(self.my_img)
#                for uut in range(0,len(self.my_img)):
#                    self.my_img[uut].destroy()
#            except AttributeError:
#                pass
            self.my_img = []
            
            for j in range(0,len(x)):
#                self.my_img = Label(self,background='#555555')#, image=render)
#                self.my_img.image = self.allimages[self.ccluster[j]]                
                render = ImageTk.PhotoImage(self.loaded_imgs[self.ccluster[j]])
                self.my_img.append([])
                self.my_img[j] = ttk.Label(background='#555555')   
                self.my_img[j].image = render                        
                
                #image_.append(my_img)
                row_ = math.floor(j/self.num_im_row) #determines the row row the image should be displayed on
                column_ = j%self.num_im_row #determines the column the image should be displayed in
                #image_[j].grid(row = row_, column = column_)
                #self.imagex.append(self.c.create_image(column_*(self.imsize + self.image_distance)+ (self.imsize / 2),row_*(self.imsize + self.image_distance)+ (self.imsize / 2), image =self.allimages[self.ccluster[j]], tags =self.ccluster[j])) #displays the image, and adds it to imagex, the list with currently displayed images
                self.imagex.append(self.c.create_image(column_*(self.imsize + self.image_distance)+ (self.imsize / 2),row_*(self.imsize + self.image_distance)+ (self.imsize / 2), image = render, tags =self.ccluster[j])) #displays the image, and adds it to imagex, the list with currently displayed images
                
                if self.bucketDisp is not 1:
                    if int(self.ccluster[j]) in [xx for vv in self.theBuckets.values() for xx in vv]: #checks if an image is already in the buckets. If so, a cyan square will be drawn around it.
                        self.greentangles.append(self.c.create_rectangle(column_*(self.imsize + self.image_distance), row_*(self.imsize + self.image_distance), column_*(self.imsize + self.image_distance)+(self.imsize + self.image_distance)-self.image_distance, row_*(self.imsize + self.image_distance)+(self.imsize + self.image_distance)-self.image_distance, outline='cyan2',width=self.rectanglewidth,tags = self.ccluster[j]))
                if self.subcyes == 1:
                    if int(self.ccluster[j]) in self.sel_im:
                        self.purpletangles.append(self.c.create_rectangle(column_*(self.imsize + self.image_distance), row_*(self.imsize + self.image_distance), column_*(self.imsize + self.image_distance)+(self.imsize + self.image_distance)-self.image_distance, row_*(self.imsize + self.image_distance)+(self.imsize + self.image_distance)-self.image_distance, outline='blueviolet',width=self.rectanglewidth,tags = self.ccluster[j]))
        
        if self.preloaded == 3:
#            try: #### this fixes a memory leak by deleting currently loaded images in the memory.
#                for uut in range(0,len(self.my_img)):
#                    self.my_img[uut].destroy()
#            except AttributeError:
#                print('watnuweer')
            self.my_img = []
            load = Image.open(self.loaded_imgs)
            for j in range(0,len(x)):
                im_getx = self.ccluster[j]%650*100
                im_gety = math.floor(self.ccluster[j]/650)*100
                
                ximg = load.crop((im_getx,im_gety,im_getx+100,imget_y+100))
                render = ImageTk.PhotoImage(ximg)
                self.my_img.append([])
                self.my_img[j] = ttk.Label(background='#555555')   
                self.my_img[j].image = render                        
                
                #image_.append(my_img)
                row_ = math.floor(j/self.num_im_row) #determines the row row the image should be displayed on
                column_ = j%self.num_im_row #determines the column the image should be displayed in
                #image_[j].grid(row = row_, column = column_)
                #self.imagex.append(self.c.create_image(column_*(self.imsize + self.image_distance)+ (self.imsize / 2),row_*(self.imsize + self.image_distance)+ (self.imsize / 2), image =self.allimages[self.ccluster[j]], tags =self.ccluster[j])) #displays the image, and adds it to imagex, the list with currently displayed images
                self.imagex.append(self.c.create_image(column_*(self.imsize + self.image_distance)+ (self.imsize / 2),row_*(self.imsize + self.image_distance)+ (self.imsize / 2), image = render, tags =self.ccluster[j])) #displays the image, and adds it to imagex, the list with currently displayed images
                if self.bucketDisp is not 1:
                    if int(self.ccluster[j]) in [xx for vv in self.theBuckets.values() for xx in vv]: #checks if an image is already in the buckets. If so, a cyan square will be drawn around it.
                        self.greentangles.append(self.c.create_rectangle(column_*(self.imsize + self.image_distance), row_*(self.imsize + self.image_distance), column_*(self.imsize + self.image_distance)+(self.imsize + self.image_distance)-self.image_distance, row_*(self.imsize + self.image_distance)+(self.imsize + self.image_distance)-self.image_distance, outline='cyan2',width=self.rectanglewidth,tags = self.ccluster[j]))
                if self.subcyes == 1:
                    if int(self.ccluster[j]) in self.sel_im:
                        self.purpletangles.append(self.c.create_rectangle(column_*(self.imsize + self.image_distance), row_*(self.imsize + self.image_distance), column_*(self.imsize + self.image_distance)+(self.imsize + self.image_distance)-self.image_distance, row_*(self.imsize + self.image_distance)+(self.imsize + self.image_distance)-self.image_distance, outline='blueviolet',width=self.rectanglewidth,tags = self.ccluster[j]))
            
        if self.preloaded == 4:
            self.my_img = []
            
            for j in range(0,4900):
                self.communication_label.configure(text='Processing image '+ str(j) + ' of ' + str(len(self.im_list)))
                #                    self.communication_label['background'] = '#99CCFF'
                self.communication_label.update()
                
                
                
                load = Image.open(self.im_list[j])
                try:
                    for orientation in ExifTags.TAGS.keys():
                        if ExifTags.TAGS[orientation]=='Orientation':
                            break
                    exif=dict(load._getexif().items())
                
                    if exif[orientation] == 3:
                        load=load.rotate(180, expand=True)
                    elif exif[orientation] == 6:
                        load=load.rotate(270, expand=True)
                    elif exif[orientation] == 8:
                        load=load.rotate(90, expand=True)
                except AttributeError:
                    pass
                except KeyError:
                    pass
                load = load.resize((self.imsize,self.imsize))
                render = ImageTk.PhotoImage(load)                
                self.my_img.append([])
                self.my_img[j] = Label(self.c,background='#555555')
                self.my_img[j].image = render                      
                row_ = math.floor(j/self.num_im_row)
                column_ = j%self.num_im_row
                self.imagex.append(self.c.create_image(column_*(self.imsize + self.image_distance)+ (self.imsize / 2),row_*(self.imsize + self.image_distance)+ (self.imsize / 2), image = render, tags =j))
                self.c.update()
                
            
        else: #same as above, but for when images are not preloaded
            try: #### this fixes a memory leak by deleting currently loaded images in the memory.
                for uut in range(0,len(self.my_img)):
                    self.my_img[uut].destroy()
            except AttributeError:
                pass
            self.my_img = []
            
            for j in range(0,len(x)):
                load = Image.open(x[j])
                try:
                    for orientation in ExifTags.TAGS.keys():
                        if ExifTags.TAGS[orientation]=='Orientation':
                            break
                    exif=dict(load._getexif().items())
                
                    if exif[orientation] == 3:
                        load=load.rotate(180, expand=True)
                    elif exif[orientation] == 6:
                        load=load.rotate(270, expand=True)
                    elif exif[orientation] == 8:
                        load=load.rotate(90, expand=True)
                except AttributeError:
                    pass
                except KeyError:
                    pass
                load = load.resize((self.imsize,self.imsize))
                render = ImageTk.PhotoImage(load)                
                self.my_img.append([])
                self.my_img[j] = Label(self.c,background='#555555')
                self.my_img[j].image = render                      
                row_ = math.floor(j/self.num_im_row)
                column_ = j%self.num_im_row
                self.imagex.append(self.c.create_image(column_*(self.imsize + self.image_distance)+ (self.imsize / 2),row_*(self.imsize + self.image_distance)+ (self.imsize / 2), image = render, tags =self.ccluster[j]))
                self.c.update()
#                self.my_img[j].destroy()
                if self.bucketDisp is not 1:
                    if int(self.ccluster[j]) in [xx for vv in self.theBuckets.values() for xx in vv]:
                            self.greentangles.append(self.c.create_rectangle(column_*(self.imsize + self.image_distance), row_*(self.imsize + self.image_distance), column_*(self.imsize + self.image_distance)+(self.imsize + self.image_distance)-self.image_distance, row_*(self.imsize + self.image_distance)+(self.imsize + self.image_distance)-self.image_distance, outline='cyan2',width=self.rectanglewidth,tags = self.ccluster[j]))                    
                if self.subcyes == 1:
                    if int(self.ccluster[j]) in self.sel_im:
                        self.purpletangles.append(self.c.create_rectangle(column_*(self.imsize + self.image_distance), row_*(self.imsize + self.image_distance), column_*(self.imsize + self.image_distance)+(self.imsize + self.image_distance)-self.image_distance, row_*(self.imsize + self.image_distance)+(self.imsize + self.image_distance)-self.image_distance, outline='blueviolet',width=self.rectanglewidth,tags = self.ccluster[j]))
        self.subcyes = 0    

#        try: #### this fixes a memory leak by deleting currently loaded images in the memory.
#            self.imagex[0].destroy()
#            for uut in range(0,len(self.my_img)):
#                self.my_img[uut].destroy()
#            for uut in range(0,len(self.greentangles)):
#                self.greentangles[uut].destroy()
#            for uut in range(0,len(self.purpletangles)):
#                self.purpletangles[uut].destroy()
#        except AttributeError:
#            pass

                
        
        #print(len(self.my_img.image_names()))
            
            
#        print('render ' + str(deep_getsizeof(render,set())))
#        print('load ' + str(deep_getsizeof(load,set())))
#        print('my_img ' + str(deep_getsizeof(self.my_img,set())))
#        print('self.ccluster ' + str(deep_getsizeof(self.ccluster,set())))
#        print('x ' + str(deep_getsizeof(x,set())))
#        print('greentangles ' + str(deep_getsizeof(self.greentangles,set())))
#        print('imagex ' + str(deep_getsizeof(self.imagex,set())))

    def _on_mousewheel(self, event):
        self.c.yview_scroll((int(-1*event.delta/120)), "units")

    #function to preload all the images. If your image collection is too large, you may run into memory issues
    def preload(self):
        self.my_img = []
        self.allimages = []
        self.loaded_imgs = []
        import time
        for z in range(0,len(self.im_list)):
            load = Image.open(self.im_list[z])
            try:            
                for orientation in ExifTags.TAGS.keys():
                    if ExifTags.TAGS[orientation]=='Orientation':
                        break
                exif=dict(load._getexif().items())
            
                if exif[orientation] == 3:
                    load=load.rotate(180, expand=True)
                elif exif[orientation] == 6:
                    load=load.rotate(270, expand=True)
                elif exif[orientation] == 8:
                    load=load.rotate(90, expand=True)                
            except AttributeError:
                pass
            except KeyError:
                pass
    
            load = load.resize((self.imsize,self.imsize))
            self.loaded_imgs.append(load)
            
            #self.allimages.append(ImageTk.PhotoImage(self.loaded_imgs[z]))            
#            try:
#                for uug in range(0,len(self.allimages)):
#                    self.allimages[uug].destroy()
#            except AttributeError:
#                pass
            
        self.preloaded = 1

    def preload2(self):
        img_map = map(Image.open, self.im_list)
        
        #total_width = 100*len(self.im_list)
        #total_height = 100
        new_im = Image.new('RGB',(65000,65000))
        
        x_offset = 0
        y_offset = 0
        for im in img_map:
            new_im.paste(im.resize((100,100)),(x_offset,y_offset))
            while x_offset < 64900:
                x_offset += 100
            else:
                x_offset = 0
                y_offset += 100
            
        self.loaded_imgs = io.BytesIO()
        new_im.save(self.loaded_imgs,format='png')
        
        self.preloaded = 3
        
        
    #function to update the image size if the user changes it
    def get_imsize(self):
        self.imsize = self.imsize_entry.get()
        try:
            self.imsize = int(self.imsize)
            if int(self.imsize) > 9:
                self.communication_label.configure(text='Image size set to ' + str(self.imsize))
                self.communication_label['background'] = '#FFFFFF'
                self.num_im_row = math.floor(self.screen_width / (self.imsize + self.image_distance))
                return True
            else:
                self.communication_label.configure(text='Image size needs to be 10 or higher')
                self.communication_label['background'] = '#FE9696'
                self.imsize = 10
                return True
            
        except ValueError:
            self.communication_label.configure(text='Please enter an integer!')
            self.communication_label['background'] = '#FE9696'
            return False

    #function to delete selected images from a bucket
    def click_del(self):
        if len(self.BucketSel) == 1:
            self.catList = self.categories.get(0,END)
            if self.bucketDisp == 1:
                for g in range(0,len(self.selected_images)):
                    self.c.delete(self.imagex[self.selected_images[g]])
                    self.current_bucket[self.selected_images[g]] = -1
                self.current_bucket = np.array(self.current_bucket)
                self.current_bucket = self.current_bucket[self.current_bucket > -1]
                self.theBuckets[self.catList[int(self.BucketSel[0])]] = self.current_bucket


    #function to focus on a selected image. From here you can draw a square to select a part of an image to compare against all other images                   
    def focus_image(self):
        if len(self.selected_images) == 1:
            self.c['scrollregion'] = (0,0,0,800)
            self.squares = None
            im_tag = self.c.gettags(self.imagex[self.selected_images[0]])
            self.bucketDisp = 5
            self.c.delete("all")
            self.focused_image = []
            d = self.im_list[int(im_tag[0])]
            self.focused_image.append(d)
            for j in range(0,len(self.focused_image)):
                load = Image.open(self.focused_image[j])
                try:
                    for orientation in ExifTags.TAGS.keys():
                        if ExifTags.TAGS[orientation]=='Orientation':
                            break
                    exif=dict(load._getexif().items())
                
                    if exif[orientation] == 3:
                        load=load.rotate(180, expand=True)
                    elif exif[orientation] == 6:
                        load=load.rotate(270, expand=True)
                    elif exif[orientation] == 8:
                        load=load.rotate(90, expand=True)                
                except AttributeError:
                    pass
                except KeyError:
                    pass

                load = load.resize((800,800))
                render = ImageTk.PhotoImage(load)
            # labels can be text or images
                my_img = Label(self,background='#555555')#, image=render)
                my_img.image = render
                #my_img.grid()
                #image_.append(my_img)
                row_ = math.floor(j/self.num_im_row)
                column_ = j%self.num_im_row
                #image_[j].grid(row = row_, column = column_)
                self.imagex.append(self.c.create_image(column_*(800 + self.image_distance)+ (800 / 2),row_*(800 + self.image_distance)+ (800 / 2), image =render,tags=im_tag))
                #if int(self.rank_list[j]) in self.theBuckets.values():
                #    self.greentangles.append(self.c.create_rectangle(column_*(self.imsize + self.image_distance), row_*(self.imsize + self.image_distance), column_*(self.imsize + self.image_distance)+(self.imsize + self.image_distance), row_*(self.imsize + self.image_distance)+(self.imsize + self.image_distance), outline='cyan2',width=5,tags = self.selected_images[j]))
    #Same function as rank images, but with a button on the GUI
    def rank_image(self):
        if len(self.selected_images) == 1:
            #self.c['scrollregion'] = (0,0,0,600)
            #self.squares = None
            im_tag = self.c.gettags(self.imagex[self.selected_images[0]])
            self.bucketDisp = 2
    
            im_tag = int(float(im_tag[0]))
            self.rank_list = self.cm[:,int(im_tag)] # get all the correlations between the selected image and all the other images
            temp_list = np.sort(self.rank_list,0)[::-1] #sorts the correlations
            self.rank_list = np.argsort(self.rank_list,0)[::-1] #sorts the id of all the images based on correlation
            self.c.xview_moveto(self.origX)  #####
            self.c.yview_moveto(self.origY) ######
            self.rank_list = np.asarray(self.rank_list)
            temp_list = np.asarray(temp_list)
            temp_list[np.isnan(temp_list)] = -100   #some images have no correlations due to issues with loading and extracting the image features. Ususally means the image file is damaged
            self.rank_list = self.rank_list[temp_list>-50] # this removes all the broken images
            self.rank_list = np.append(im_tag,self.rank_list) # this adds the selected (queried) image to the image list to be displayed
            #self.bucketDisp = 0
            self.c.delete("all")
            self.display_images(self.rank_list) #function to display the ranked list of images. By default it displays all images, but may need to be limited for large datasets

            self.communication_label['background'] = '#FFFFFF'
            self.communication_label.configure(text='Finished calculating. Showing the ranking')
            plt.close("all")

    def rank_bucket(self):
        if self.bucketDisp == 1:
            self.bucketDisp = 0
            bucket_cm = self.cm[self.current_bucket]
            bucket_cm2 = bucket_cm.reshape((-1,1))
            super_index = np.arange(0,len(self.im_list))
            super_index = np.resize(super_index,len(super_index)*len(self.current_bucket))
            #super_index = super_index[np.flip(np.argsort(np.nan_to_num(bucket_cm2),0))]
            self.super_index = pd.unique(super_index[np.flip(np.argsort(np.nan_to_num(bucket_cm2),0))].squeeze())
            
            self.c.xview_moveto(self.origX)  #####
            self.c.yview_moveto(self.origY) ######
            
            self.c.delete("all")
            self.display_images(self.super_index) #function to display the ranked list of images. By default it displays all images, but may need to be limited for large datasets

            self.communication_label['background'] = '#FFFFFF'
            self.communication_label.configure(text='Finished calculating. Showing the ranking')



    # This function shows a filtered selection of buckets.                   
    def filter_buckets(self):
#        filter_in = self.categories2.curselection() #acquire the selected buckets to show
#        print(filter_in)        
#        filter_out = self.categories3.curselection() #acquire the selected buckets not to show
#        print(filter_out)
        def find_dubs(L):
            seen = set()
            seen2 = set()
            seen_add = seen.add
            seen2_add = seen2.add
            for item in L:
                if item in seen:
                    seen2_add(item)
                else:
                    seen_add(item)
            return list(seen2)
        
        self.cluster_in = []
        self.cluster_out = []
        for i in range(0,len(self.filter_in)):
                self.cluster_in.append(self.theBuckets[self.categories.get(self.filter_in[i])])
        self.cluster_in = list(chain.from_iterable(self.cluster_in))
        if len(self.filter_in)> 1:
            self.cluster_in = find_dubs(self.cluster_in)
        try:
            len(self.filter_out)
        except AttributeError:
            self.filter_out = []
        for i in range(0,len(self.filter_out)):
                self.cluster_out.append(self.theBuckets[self.categories.get(self.filter_out[i])])
        self.cluster_out = list(chain.from_iterable(self.cluster_out)) 
        self.filtered_bucket = np.setdiff1d(self.cluster_in,self.cluster_out)
        self.imagex = []
        self.c.delete("all")
        self.display_images(self.filtered_bucket)
        self.filter_in = []
        self.filter_out = []
            #self.cluster = self.cluster[np.nonzero(self.cluster)]
 
    
    # This function compares the drawn square to all other images                   
    def query_selection(self):
        load = Image.open(self.focused_image[0])
        try:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation]=='Orientation':
                    break
            exif=dict(load._getexif().items())
        
            if exif[orientation] == 3:
                load=load.rotate(180, expand=True)
            elif exif[orientation] == 6:
                load=load.rotate(270, expand=True)
            elif exif[orientation] == 8:
                load=load.rotate(90, expand=True)                
        except AttributeError:
            pass
        except KeyError:
            pass

        width, height = load.size
        evex1 = min(self.evex1,self.evex2)
        evex2 = max(self.evex1,self.evex2)
        evey1 = min(self.evey1,self.evey2)
        evey2 = max(self.evey1,self.evey2)
        
        evex1 = width / (800 / evex1)
        evex2 = width / (800 / evex2)
        evey1 = height / (800 / evey1)
        evey2 = height / (800 / evey2)
        

        def feature_extraction2(neural_net, im_list,evex1,evex2,evey1,evey2):
            f = im_list
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            if neural_net == 'inception_v3':
                f_size = 2048
                model = models.inception_v3(pretrained='imagenet')
                layer = model._modules.get('Mixed_7c')
            elif neural_net == 'resnet152': #2084
                f_size = 2048
                model = models.resnet152(pretrained=True)
                layer = model._modules.get('avgpool')
            elif neural_net == 'resnet18': #512
                f_size =512
                model = models.resnet18(pretrained=True)
                layer = model._modules.get('avgpool')
            elif neural_net == 'vgg16': #4096
                f_size = 4096
                model = models.vgg16(pretrained=True)
                model.classifier = model.classifier[:-1]
                layer   = 'nothing'
            elif neural_net == 'vgg19': #4096
                f_size =    4096
                model = models.vgg19(pretrained=True)
                model.classifier = model.classifier[:-1]
                layer = 'nothing'
            elif neural_net == 'densenet161': #2208
                f_size =2208
                model = models.densenet161(pretrained=True)	
                model = model.features
            elif neural_net == 'squeezenet1_0': #512
                f_size = 1000
                model = models.squeezenet1_0(pretrained=True)
                #model = model.features
                layer = 'nothing'
            elif neural_net == 'alexnet':    
                f_size = 4096
                model = models.alexnet(pretrained=True)
                model.classifier = model.classifier[:-1]
                layer = 'nothing'
            model.eval()
            model = model.to(device)
            mean=[0.485, 0.456, 0.406]
            std=[0.229, 0.224, 0.225]
            if neural_net == 'inception_v3':
                transform = transforms.Compose([
                            transforms.Resize((299,299)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std)
                        ])
            else:
                transform = transforms.Compose([
                            transforms.Resize((224,224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std)
                        ])
            def get_vector2(image_name,f_size, layer, transform, model, neural_net,evex1,evex2,evey1,evey2):
                img = Image.open(image_name)
                try:
                    for orientation in ExifTags.TAGS.keys():
                        if ExifTags.TAGS[orientation]=='Orientation':
                            break
                    exif=dict(img._getexif().items())
                
                    if exif[orientation] == 3:
                        img=img.rotate(180, expand=True)
                    elif exif[orientation] == 6:
                        img=img.rotate(270, expand=True)
                    elif exif[orientation] == 8:
                        img=img.rotate(90, expand=True)                
                except AttributeError:
                    pass
                except KeyError:
                    pass
                img = img.crop((evex1,evey1,evex2,evey2))
                
                if img.mode == 'RGB':
                    try:
                        t_img = transform(img).unsqueeze(0)
                    except OSError:
                        t_img = transform(img).unsqueeze(0)
                    t_img = t_img.to(device)
                    if neural_net == 'alexnet' or neural_net =='vgg19' or neural_net =='vgg16' or neural_net =='alexnet' or neural_net =='squeezenet1_0':
                        torch.cuda.empty_cache()
                        my_embeddingz = model(t_img)
                        my_embeddingz = my_embeddingz.cpu()
                        my_embeddingz = np.ndarray.flatten(my_embeddingz.data.numpy())
                    elif neural_net == 'densenet161':
                        featuresY = model(t_img)
                        my_embeddingz = F.relu(featuresY,inplace= True)
                        my_embeddingz = F.avg_pool2d(my_embeddingz, kernel_size=7, stride=1).view(featuresY.size(0), -1)
                        my_embeddingz = my_embeddingz.cpu()
                        my_embeddingz = np.ndarray.flatten(my_embeddingz.data.numpy())
                    elif neural_net == 'inception_v3':
                        my_embeddingz = torch.zeros((1,f_size,8,8))
                        my_embeddingz = my_embeddingz.to(device)
                            # 4. Define a function that will copy the output of a layer
                        def copy_data(m, i, o):
                            my_embeddingz.copy_(o.data)
                            # 5. Attach that function to our selected layer
                        h = layer.register_forward_hook(copy_data)
                            # 6. Run the model on our transformed image
                        model(t_img)
                        #    # 7. Detach our copy function from the layer
                        h.remove()
                        my_embeddingz = F.avg_pool2d(my_embeddingz, kernel_size=8)
                        my_embeddingz = my_embeddingz.view(my_embeddingz.size(0), -1)
                        my_embeddingz = my_embeddingz.cpu()
                        my_embeddingz = np.ndarray.flatten(my_embeddingz.data.numpy())
                    else:
                        my_embeddingz = torch.zeros((1,f_size,1,1))
                        my_embeddingz = my_embeddingz.to(device)
                            # 4. Define a function that will copy the output of a layer
                        def copy_data(m, i, o):
                            my_embeddingz.copy_(o.data)
                            # 5. Attach that function to our selected layer
                        h = layer.register_forward_hook(copy_data)
                            # 6. Run the model on our transformed image
                        model(t_img)
                        #    # 7. Detach our copy function from the layer
                        h.remove()
                        my_embeddingz = my_embeddingz.cpu()
                        my_embeddingz = np.ndarray.flatten(my_embeddingz.data.numpy())
                else:
                    my_embeddingz = np.zeros((f_size,))
                return my_embeddingz
            my_embed = []
            self.progress["value"] = 0
            self.progress["maximum"] = len(f)-1
            for i in range(0,len(f)):
                my_embed.append(get_vector2(f[i],f_size,layer,transform,model,neural_net,evex1,evex2,evey1,evey2))

            features = np.asarray(my_embed)
            return features


        self.focusfeatures = feature_extraction2(self.neuralnet,self.focused_image,evex1,evex2,evey1,evey2)

        def create_matrix2(focusfeatures,features,distance_metric):
            focusfeatures = np.squeeze(np.expand_dims(focusfeatures,0))
            features = np.squeeze(np.expand_dims(features,0))
            features_t = np.transpose(features)
            focusfeatures_t = np.transpose(focusfeatures)
            cm = []                
            sumX = sum(features_t)
            focussumX = sum(focusfeatures_t)
            sumsquareX = sum(features_t**2)
            
#                for i in range(0,features.shape[0]):
            feat0 = focusfeatures
            sumXY = np.dot(feat0,features_t)
            r = features.shape[1]*sumXY - focussumX*sumX 
            s = ((features.shape[1] * sumsquareX) - sumX**2)
            t = 1./((s[0]*s)**0.5)
            u = r * t
            cm.append(u)
            cm = np.asmatrix(cm)
            return cm
        self.focuscm = create_matrix2(self.focusfeatures,self.features,'correlation')
        temp_list = np.sort(self.focuscm.transpose(),0)[::-1]
        
        self.focus_list = np.argsort(self.focuscm.transpose(),0)[::-1]
        #self.rank_list = self.rank_list[0:1000]
        self.c.xview_moveto(self.origX)  #####
        self.c.yview_moveto(self.origY) ######
        self.focus_list = np.asarray(self.focus_list)
        temp_list = np.asarray(temp_list)
        temp_list[np.isnan(temp_list)] = -100
        self.focus_list = self.focus_list[temp_list>-50]
        self.bucketDisp = 0
        self.c.delete("all")
        x = []
        for i in range(0,len(self.focus_list)):
            d = self.im_list[self.focus_list[i]]
            x.append(d)
        self.display_images(self.focus_list)



    #Clusters the currently displayed images into subclusters per the selected images by the user
    def subcluster(self):
        m = len(self.selected_images) #number of images selected (number of subclusters)
        im_tags = [] 
        for ts in range(len(self.imagex)):
            im_tags.append(int(self.c.gettags(self.imagex[ts])[0])) #all images currently displayed
        im_tags = np.asarray(im_tags)
        self.sel_im = im_tags[self.selected_images] #id of images currently selected
        #sel_im = sel_im.to_numpy()
        cur_cm = self.cm[im_tags] #correlations of images currently selected
        cur_cm = cur_cm[:,self.sel_im] #correlations of images currently selected
        cur_cm = np.nan_to_num(cur_cm) #removes nans
        cur_cmax = np.argmax(cur_cm,1) #determine which selected image correlates higher with each image
        nco = np.array([],dtype='int') #empty array for subclustered images
        for kk in range(m): #going through the selected images
            tb = im_tags[cur_cmax==kk]
            tc = cur_cm[cur_cmax==kk][:,kk]
            
            tb = tb[np.flip(np.argsort(tc))]
            tb = np.delete(tb,np.intersect1d(tb,self.sel_im,return_indices=True)[1],0)
            try:
                #nco= np.hstack([nco,sel_im[kk],self.currentcluster[cur_cmax==kk]])
                nco= np.hstack([nco,self.sel_im[kk],tb])

            except ValueError:
                #nco= np.vstack([nco,sel_im[kk],self.currentcluster[cur_cmax==kk]])
                nco= np.vstack([nco,self.sel_im[kk],tb])
        #for overview:
        try:
            self.xsorted
        except AttributeError:
            self.xsorted = np.argsort(im_tags)
            self.xsortedimtags = im_tags[self.xsorted]
                            
        ypos = np.searchsorted(self.xsortedimtags, nco)
        print(nco)
        print(ypos)
        print(self.xsorted)
        self.ind_for_overview = self.xsorted[ypos]
        ###
        self.subcyes = 1
        self.display_images(nco)
        
    def subclusterkmeans(self):
        m = len(self.selected_images)
        im_tags = [] 
        for ts in range(len(self.imagex)):
            im_tags.append(int(self.c.gettags(self.imagex[ts])[0]))
        im_tags = np.asarray(im_tags)
        self.sel_im = im_tags[self.selected_images]
        
        
        def color_features(an_image):
            an_image = an_image.convert('HSV')    
            imarray = np.asarray(an_image)
            imarray = imarray.reshape(imarray.shape[0]*imarray.shape[1],3)
            reimarray = np.vstack((imarray[:,0]/255*360,imarray[:,1]/255,imarray[:,2]/255)).transpose()
            reimarray[:,0][reimarray[:,0] < 20] = 0
            reimarray[:,0][np.where((reimarray[:,0] >= 20) & (reimarray[:,0] <= 40))] = 1
            reimarray[:,0][np.where((reimarray[:,0] >= 40) & (reimarray[:,0] <= 75))] = 2
            reimarray[:,0][np.where((reimarray[:,0] >= 75) & (reimarray[:,0] <= 155))] = 3
            reimarray[:,0][np.where((reimarray[:,0] >= 155) & (reimarray[:,0] <= 190))] = 4
            reimarray[:,0][np.where((reimarray[:,0] >= 190) & (reimarray[:,0] <= 270))] = 5
            reimarray[:,0][np.where((reimarray[:,0] >= 270) & (reimarray[:,0] <= 295))] = 6
            reimarray[:,0][reimarray[:,0] > 295] = 7
            
            reimarray[:,1][reimarray[:,1] < 0.2] = 0
            reimarray[:,1][np.where((reimarray[:,1] >= 0.2) & (reimarray[:,1] <= 0.7))] = 1
            reimarray[:,1][reimarray[:,1] > 0.7] = 2
            
            reimarray[:,2][reimarray[:,2] < 0.2] = 0
            reimarray[:,2][np.where((reimarray[:,2] >= 0.2) & (reimarray[:,2] <= 0.7))] = 1
            reimarray[:,2][reimarray[:,2] > 0.7] = 2
        
            colorvector = reimarray[:,0] * 9 + reimarray[:,1] * 3 +reimarray[:,2]
            colorvector = np.histogram(colorvector,76)[0]
            return colorvector
        def get_colors(image_name):
            img = Image.open(image_name)
            color_feature = color_features(img)
            return color_feature
#        my_colors = []
#        for i in range(len(im_tags)):
#            my_colors.append(get_colors(self.im_list[im_tags[i]]))
#        my_colors = np.asarray(my_colors)
        features = self.features[im_tags]
#        features = features/np.max(features)
#        features = np.hstack((features,my_colors))
#        features = my_colors
        centers = features[np.isin(im_tags,self.sel_im)]
        #sel_im = sel_im.to_numpy()
        ward = KMeans(n_clusters=len(self.sel_im), init=centers, n_init=1).fit(features)
        test2 = ward.labels_
#        cluster_ind = []
#        for i in range(0,len(sel_im)):
#            cluster_ind.append([])
#        for k in range(0,len(test2)):
#            cluster_ind[test2[k]].append(k)
#        xlength = []
#        for q in range(0,len(cluster_ind)):
#            xlength.append(len(cluster_ind[q]))
#        cluster_indX = np.zeros((max(xlength),len(cluster_ind)))-1        
#        for r in range(0,len(cluster_ind)):
#            for s in range(0,len(cluster_ind[r])):
#                cluster_indX[s,r] = int(cluster_ind[r][s])
#        cluster_indX = cluster_indX.astype(int)
        


#        cur_cm = self.cm[im_tags]
#        cur_cm = cur_cm[:,sel_im]
#        cur_cm = np.nan_to_num(cur_cm)
#        cur_cmax = np.argmax(cur_cm,1)
        nco = np.array([],dtype='int')
        for kk in range(m):
            tb = im_tags[test2==kk]#.to_numpy()
            try:
                #nco= np.hstack([nco,sel_im[kk],self.currentcluster[cur_cmax==kk]])
                nco= np.hstack([nco,self.sel_im[kk],tb])

            except ValueError:
                #nco= np.vstack([nco,sel_im[kk],self.currentcluster[cur_cmax==kk]])
                nco= np.vstack([nco,self.sel_im[kk],tb])
        self.subcyes = 1
        self.display_images(nco)


    def purtangles(self):
        self.purpletangles = []
        self.sel_im
        self.purpletangles.append(self.c.create_rectangle(column_*(self.imsize + self.image_distance), row_*(self.imsize + self.image_distance), column_*(self.imsize + self.image_distance)+(self.imsize + self.image_distance)-self.image_distance, row_*(self.imsize + self.image_distance)+(self.imsize + self.image_distance)-self.image_distance, outline='cyan2',width=self.rectanglewidth,tags = self.ccluster[j]))
        
    def query_external(self):
        self.external_image = str(filedialog.askopenfilename())
            
#        self.external_image = Image.open(self.external_image)
#        try:
#            for orientation in ExifTags.TAGS.keys():
#                if ExifTags.TAGS[orientation]=='Orientation':
#                    break
#            exif=dict(self.external_imaged._getexif().items())
#        
#            if exif[orientation] == 3:
#                self.external_image=self.external_image.rotate(180, expand=True)
#            elif exif[orientation] == 6:
#                self.external_image=self.external_image.rotate(270, expand=True)
#            elif exif[orientation] == 8:
#                self.external_image=self.external_image.rotate(90, expand=True)                
#        except AttributeError:
#            pass
#        except KeyError:
#            pass
#        width, height = load.size
#        evex1 = min(self.evex1,self.evex2)
#        evex2 = max(self.evex1,self.evex2)
#        evey1 = min(self.evey1,self.evey2)
#        evey2 = max(self.evey1,self.evey2)
#        
#        evex1 = width / (800 / evex1)
#        evex2 = width / (800 / evex2)
#        evey1 = height / (800 / evey1)
#        evey2 = height / (800 / evey2)
        

        def feature_extraction3(neural_net, im_list):
            f = im_list
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            if neural_net == 'inception_v3':
                f_size = 2048
                model = models.inception_v3(pretrained='imagenet')
                layer = model._modules.get('Mixed_7c')
            elif neural_net == 'resnet152': #2084
                f_size = 2048
                model = models.resnet152(pretrained=True)
                layer = model._modules.get('avgpool')
            elif neural_net == 'resnet18': #512
                f_size =512
                model = models.resnet18(pretrained=True)
                layer = model._modules.get('avgpool')
            elif neural_net == 'vgg16': #4096
                f_size = 4096
                model = models.vgg16(pretrained=True)
                model.classifier = model.classifier[:-1]
                layer   = 'nothing'
            elif neural_net == 'vgg19': #4096
                f_size =    4096
                model = models.vgg19(pretrained=True)
                model.classifier = model.classifier[:-1]
                layer = 'nothing'
            elif neural_net == 'densenet161': #2208
                f_size =2208
                model = models.densenet161(pretrained=True)	
                model = model.features
            elif neural_net == 'squeezenet1_0': #512
                f_size = 1000
                model = models.squeezenet1_0(pretrained=True)
                #model = model.features
                layer = 'nothing'
            elif neural_net == 'alexnet':    
                f_size = 4096
                model = models.alexnet(pretrained=True)
                model.classifier = model.classifier[:-1]
                layer = 'nothing'
            model.eval()
            model = model.to(device)
            mean=[0.485, 0.456, 0.406]
            std=[0.229, 0.224, 0.225]
            if neural_net == 'inception_v3':
                transform = transforms.Compose([
                            transforms.Resize((299,299)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std)
                        ])
            else:
                transform = transforms.Compose([
                            transforms.Resize((224,224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std)
                        ])
            def get_vector3(image_name,f_size, layer, transform, model, neural_net):
                img = Image.open(image_name)
                try:
                    for orientation in ExifTags.TAGS.keys():
                        if ExifTags.TAGS[orientation]=='Orientation':
                            break
                    exif=dict(img._getexif().items())
                
                    if exif[orientation] == 3:
                        img=img.rotate(180, expand=True)
                    elif exif[orientation] == 6:
                        img=img.rotate(270, expand=True)
                    elif exif[orientation] == 8:
                        img=img.rotate(90, expand=True)                
                except AttributeError:
                    pass
                except KeyError:
                    pass

                #img = img.crop((evex1,evey1,evex2,evey2))
                
                if img.mode == 'RGB':
                    try:
                        t_img = transform(img).unsqueeze(0)
                    except OSError:
                        t_img = transform(img).unsqueeze(0)
                    t_img = t_img.to(device)
                    if neural_net == 'alexnet' or neural_net =='vgg19' or neural_net =='vgg16' or neural_net =='alexnet' or neural_net =='squeezenet1_0':
                        torch.cuda.empty_cache()
                        my_embeddingz = model(t_img)
                        my_embeddingz = my_embeddingz.cpu()
                        my_embeddingz = np.ndarray.flatten(my_embeddingz.data.numpy())
                    elif neural_net == 'densenet161':
                        featuresY = model(t_img)
                        my_embeddingz = F.relu(featuresY,inplace= True)
                        my_embeddingz = F.avg_pool2d(my_embeddingz, kernel_size=7, stride=1).view(featuresY.size(0), -1)
                        my_embeddingz = my_embeddingz.cpu()
                        my_embeddingz = np.ndarray.flatten(my_embeddingz.data.numpy())
                    elif neural_net == 'inception_v3':
                        my_embeddingz = torch.zeros((1,f_size,8,8))
                        my_embeddingz = my_embeddingz.to(device)
                            # 4. Define a function that will copy the output of a layer
                        def copy_data(m, i, o):
                            my_embeddingz.copy_(o.data)
                            # 5. Attach that function to our selected layer
                        h = layer.register_forward_hook(copy_data)
                            # 6. Run the model on our transformed image
                        model(t_img)
                        #    # 7. Detach our copy function from the layer
                        h.remove()
                        my_embeddingz = F.avg_pool2d(my_embeddingz, kernel_size=8)
                        my_embeddingz = my_embeddingz.view(my_embeddingz.size(0), -1)
                        my_embeddingz = my_embeddingz.cpu()
                        my_embeddingz = np.ndarray.flatten(my_embeddingz.data.numpy())
                    else:
                        my_embeddingz = torch.zeros((1,f_size,1,1))
                        my_embeddingz = my_embeddingz.to(device)
                            # 4. Define a function that will copy the output of a layer
                        def copy_data(m, i, o):
                            my_embeddingz.copy_(o.data)
                            # 5. Attach that function to our selected layer
                        h = layer.register_forward_hook(copy_data)
                            # 6. Run the model on our transformed image
                        model(t_img)
                        #    # 7. Detach our copy function from the layer
                        h.remove()
                        my_embeddingz = my_embeddingz.cpu()
                        my_embeddingz = np.ndarray.flatten(my_embeddingz.data.numpy())
                else:
                    my_embeddingz = np.zeros((f_size,))
                return my_embeddingz
            my_embed = []
            self.progress["value"] = 0
            self.progress["maximum"] = len(f)-1
            
            my_embed.append(get_vector3(f,f_size,layer,transform,model,neural_net))

            features = np.asarray(my_embed)
            return features


        self.externalfeatures = feature_extraction3(self.neuralnet,self.external_image)

        def create_matrix3(focusfeatures,features,distance_metric):
            focusfeatures = np.squeeze(np.expand_dims(focusfeatures,0))
            features = np.squeeze(np.expand_dims(features,0))
            features_t = np.transpose(features)
            focusfeatures_t = np.transpose(focusfeatures)
            cm = []                
            sumX = sum(features_t)
            focussumX = sum(focusfeatures_t)
            sumsquareX = sum(features_t**2)
            
#                for i in range(0,features.shape[0]):
            feat0 = focusfeatures
            sumXY = np.dot(feat0,features_t)
            r = features.shape[1]*sumXY - focussumX*sumX 
            s = ((features.shape[1] * sumsquareX) - sumX**2)
            t = 1./((s[0]*s)**0.5)
            u = r * t
            cm.append(u)
            cm = np.asmatrix(cm)
            return cm
        self.focuscm = create_matrix3(self.externalfeatures,self.features,'correlation')
        temp_list = np.sort(self.focuscm.transpose(),0)[::-1]
        
        self.focus_list = np.argsort(self.focuscm.transpose(),0)[::-1]
        #self.rank_list = self.rank_list[0:1000]
        self.c.xview_moveto(self.origX)  #####
        self.c.yview_moveto(self.origY) ######
        self.focus_list = np.asarray(self.focus_list)
        temp_list = np.asarray(temp_list)
        temp_list[np.isnan(temp_list)] = -100
        self.focus_list = self.focus_list[temp_list>-50]
        self.bucketDisp = 0
        self.c.delete("all")
        x = []
        for i in range(0,len(self.focus_list)):
            d = self.im_list[self.focus_list[i]]
            x.append(d)
        self.display_images(self.focus_list)


    def expand_cluster(self):
        cluster_e = self.df[self.num_clus]
        cluster_e = cluster_e[cluster_e > -1]
        current_features = np.mean(self.features[cluster_e],0)
        try:
            self.new_threshold = self.new_threshold
        except AttributeError:
                self.new_threshold = self.threshold
                
        def create_matrix3(current_features,features,distance_metric):
            current_features = np.squeeze(np.expand_dims(current_features,0))
            features = np.squeeze(np.expand_dims(features,0))
            features_t = np.transpose(features)
            current_features_t = np.transpose(current_features)
            cm = []                
            sumX = sum(features_t)
            focussumX = sum(current_features_t)
            sumsquareX = sum(features_t**2)
            
#                for i in range(0,features.shape[0]):
            feat0 = current_features
            sumXY = np.dot(feat0,features_t)
            r = features.shape[1]*sumXY - focussumX*sumX 
            s = ((features.shape[1] * sumsquareX) - sumX**2)
            t = 1./((s[0]*s)**0.5)
            u = r * t
            cm.append(u)
            cm = np.asarray(cm)
            return cm
        currentcm = np.nan_to_num(create_matrix3(current_features,self.features,'correlation').transpose())
        self.new_threshold = self.new_threshold - 0.1
        expanded_cluster = np.expand_dims(np.arange(0,len(self.im_list)),1)
        expanded_cluster = expanded_cluster[currentcm > self.new_threshold]
        expanded_cluster = np.setdiff1d(expanded_cluster,cluster_e)
        expanded_cluster = np.concatenate((expanded_cluster,cluster_e))
        self.display_images(expanded_cluster)

    # function to display the contents of the selected bucket 
    def showBucket(self):
#        with open('D:\\PhD\\Visual Analytics\\MockCase\\zForBucket.pickle', 'wb') as handle:
#            pickle.dump(self.theBuckets,handle,protocol=pickle.HIGHEST_PROTOCOL)
        self.bucketDisp = 1
        self.c.delete("all")
        selected = self.categories.curselection() #acquire the selected category
        if len(selected) == 1:
            self.BucketSel = selected
            self.cluster = self.theBuckets[self.categories.get(selected)]
            self.cluster = np.asarray(self.cluster)
            self.cluster = self.cluster[np.nonzero(self.cluster)]
            self.current_bucket = self.cluster
            num_im = len(self.cluster)
            self.communication_label.configure(text='The bucket '+ self.categories.get(selected) +' is shown. This bucket contains ' + str(num_im) + ' images.')

#            with open('D:\\PhD\\Visual Analytics\\MockCase\\zForBucket.pickle', 'wb') as handle:
#                pickle.dump(self.cluster,handle,protocol=pickle.HIGHEST_PROTOCOL)

        if len(selected) > 1:
            self.cluster = []
            bucket_com = ''
            for i in range(0,len(selected)):
                self.cluster.append(self.theBuckets[self.categories.get(selected[i])])
                bucket_com = bucket_com + self.categories.get(selected[i]) + '|' 
#            tempC = self.cluster[0]

#            for i in range(1,len(selected)):
#                tempC = set(tempC) & set(self.cluster[i])
                
#            tempC = list(tempC)
#            self.cluster = np.zeros((len(tempC),1))    
#            for i in range(0,len(tempC)):
#                self.cluster[i,0] = tempC[i]
            self.cluster = list(chain.from_iterable(self.cluster)) 
            self.cluster = list(set(self.cluster))
            #self.cluster = self.cluster[np.nonzero(self.cluster)]
            self.current_bucket = self.cluster
            num_im = len(self.cluster)
            self.communication_label.configure(text='The following buckets are shown: '+ bucket_com +'. These buckets contain a total of ' + str(num_im) + ' images.')
            
        self.imagex = []
        self.cluster = np.asarray(self.cluster)
        self.display_images(self.cluster)

    # function to add the currently displayed cluster to the selected bucket(s)
    def addCluster2bucket(self):
        self.refresh = 0
        selected = self.categories.curselection()
        bucket_com = ''
        for p in range(0, len(selected)):
            temp_c = self.theBuckets[self.categories.get(selected[p])]
            temp_c = np.asarray(temp_c)
            temp_x = np.concatenate((self.currentcluster,temp_c),axis = 0)
            temp_y = numpy_unique_ordered(temp_x)
            temp_y = np.asarray(temp_y).astype(int)
            self.theBuckets[self.categories.get(selected[p])] = temp_y
            bucket_com = bucket_com + str(self.categories.get(selected[p])) + ' '
        self.communication_label.configure(text='cluster ' + str(self.num_clus) + ' has been added to ' + bucket_com)
        
    # function to add the currently selected image(s) to the selected bucket(s)
    def addSelection2bucket(self):
        self.refresh = 0
        selected = self.categories.curselection()
        bucket_com = ''
        for p in range(0, len(selected)):
            bucket_com = bucket_com + str(self.categories.get(selected[p])) + ' '
            temp_c = np.asarray(self.theBuckets[self.categories.get(selected[p])])
#            if self.bucketDisp==0:
            temp_z = self.ccluster[self.selected_images]
            temp_x = np.concatenate((temp_z,temp_c),axis = 0)
#            elif self.bucketDisp==1:
#                temp_z = self.ccluster[self.selected_images]
#                temp_x = np.concatenate((temp_z,temp_c),axis = 0)
#            elif self.bucketDisp==2:
#                temp_z = self.ccluster[self.selected_images]
#                temp_x = np.concatenate((temp_z,temp_c),axis = 0)
            temp_y = numpy_unique_ordered(temp_x)
            temp_y = np.asarray(temp_y).astype(int)            
            self.theBuckets[self.categories.get(selected[p])] = temp_y            
        if len(temp_z) == 1:
            self.communication_label.configure(text=str(len(temp_z)) + ' image has been added to ' + bucket_com)
        else:
            self.communication_label.configure(text=str(len(temp_z)) + ' images have been added to ' + bucket_com)
            
    # function to add a new user created bucket. Buckets are sorted alphabeteically.
    def addCategory(self):
        if self.e2.get() not in self.catList:
            self.categories.insert(END, self.e2.get())
            self.theBuckets[self.e2.get()] = []
            self.catList = self.categories.get(0,END)
            self.catList = sorted(self.catList, key=str.lower)
            boxscrollbar = Scrollbar(width = 10)
            self.categories = Listbox(width=30,background='#777777',foreground='white',yscrollcommand=boxscrollbar.set,exportselection=0)
            self.categories['selectmode'] = 'extended'               
            self.categories.place(x=585,y=50)
            for k in range(0,len(self.catList)):
                self.categories.insert(END,self.catList[k])
            boxscrollbar.config(command=self.categories.yview)
            boxscrollbar.place(in_=self.categories,relx=1.0, relheight=1)
            self.categories.select_set(self.catList.index(self.e2.get()))
            self.categories.see(self.catList.index(self.e2.get()))
            self.categories.bind('<Button-1>', self.deselect_list )
    #function to display the next bucket
    def nextCluster(self):
        if self.num_clus < self.df.shape[1]-1:
            self.num_clus += 1
            #self.b1['text'] ="next cluster" + ' (' + str(self.num_clus) + ')'
            self.c.xview_moveto(self.origX)
            self.c.yview_moveto(self.origY)
            self.communication_label.configure(text='You are currently viewing cluster ' + str(self.num_clus+1) +' out of ' + str(self.df.shape[1]) + ' clusters')
            self.new_threshold = self.threshold
    
    #function to display the previous bucket
    def prevCluster(self):
        if self.num_clus > 0:
            self.num_clus -= 1
            self.c.xview_moveto(self.origX)
            self.c.yview_moveto(self.origY)
            self.communication_label.configure(text='You are currently viewing cluster ' + str(self.num_clus+1) +' out of ' + str(self.df.shape[1]) + ' clusters')
            self.new_threshold = self.threshold

    def showCluster(self):
        self.num_clus = int(self.ecluster.get())-1
        self.c.xview_moveto(self.origX)
        self.c.yview_moveto(self.origY)
        self.communication_label.configure(text='You are currently viewing cluster ' + str(self.num_clus+1) +' out of ' + str(self.df.shape[1]) + ' clusters')
        self.new_threshold = self.threshold
        

    def change_name(self):
        new_bucket = self.e2.get()
        old_bucket = self.categories.curselection()
        if len(old_bucket) == 1:
            old_bucket = self.catList[old_bucket[0]]
            self.catList = list(self.catList)
            self.catList[self.catList.index(old_bucket)] = new_bucket
            self.theBuckets[new_bucket] = self.theBuckets.pop(old_bucket)
            self.catList = sorted(self.catList, key=str.lower)
            boxscrollbar = Scrollbar(width = 10)
            self.categories = Listbox(width=30,background='#777777',foreground='white',yscrollcommand=boxscrollbar.set,exportselection=0)
            self.categories['selectmode'] = 'extended'               
            self.categories.place(x=585,y=50)
            for k in range(0,len(self.catList)):
                self.categories.insert(END,self.catList[k])
            boxscrollbar.config(command=self.categories.yview)
            boxscrollbar.place(in_=self.categories,relx=1.0, relheight=1)
            self.categories.select_set(self.catList.index(self.e2.get()))
            self.categories.see(self.catList.index(self.e2.get()))
            self.categories.bind('<Button-1>', self.deselect_list )
        
    #function to display the images of the current cluster, e.g. to switch from the bucket view back to the current cluster.
    def showImg(self):
        self.bucketDisp = 0
        self.imagex = []
        self.c.delete("all")
        num_c = self.num_clus
        #df = pd.read_csv('forAppOID.csv', header=None) 
        #df=df-1 #
        cluster = self.df[num_c]
        cluster = cluster[cluster > -1]
        self.currentcluster = cluster
        self.greentangles = []
        self.communication_label.configure(text='You are currently viewing cluster ' + str(self.num_clus+1) +' out of ' + str(self.df.shape[1]) + ' clusters. This cluster contains ' + str(len(cluster)) + ' images.')

        
        self.display_images(cluster)
    #function to save the current session as a pickle file. The buckets, the list of images, the clusters, the current cluster number, the array with correlations, and the array with features are saved.
    def save_as(self):
        self.catList = self.categories.get(0,END)        
        self.answer = filedialog.asksaveasfilename(defaultextension=".pickle")   #this will make the file path a string
            
        pickle_out = open(self.answer,"wb")
        pickle.dump([self.im_list,self.df,self.catList,self.theBuckets,self.num_clus,self.cm,self.features,self.loaded_imgs,self.X_embed],pickle_out,protocol=pickle.HIGHEST_PROTOCOL)
        pickle_out.close()
        self.communication_label.configure(text='Saved!')        
    #function to load a previous session. 
    def load_as(self):
        self.answer2 = filedialog.askopenfilename(defaultextension=".pickle")   #this will make the file path a string
        try:
            pickle_in = open(self.answer2,"rb")        
            self.im_list,self.df,self.catList,self.theBuckets,self.num_clus, self.cm, self.features, self.loaded_imgs, self.X_embed = pickle.load(pickle_in) #new version with TSNE
        except ValueError:
                pickle_in = open(self.answer2,"rb")        
                self.im_list,self.df,self.catList,self.theBuckets,self.num_clus, self.cm, self.features, self.loaded_imgs = pickle.load(pickle_in) #load old version without TSNE
            
        pickle_in.close()
        if len(self.loaded_imgs) > 0:
            self.preloaded = 1
            for z in range(len(self.loaded_imgs)):
                self.allimages.append(ImageTk.PhotoImage(self.loaded_imgs[z]))
        boxscrollbar = Scrollbar(width = 10)
        self.categories = Listbox(width=30,background='#777777',foreground='white',yscrollcommand=boxscrollbar.set,exportselection=0)
        self.categories['selectmode'] = 'extended'               
        self.categories.place(x=585,y=50)
        for k in range(0,len(self.catList)):
            self.categories.insert(END,self.catList[k])
        boxscrollbar.config(command=self.categories.yview)
        boxscrollbar.place(in_=self.categories,relx=1.0, relheight=1)
        self.categories.bind('<Button-1>', self.deselect_list )
        if len(self.im_list) > 0:
            if len(self.cm) > 0:
                if len(self.df) > 0:
                    self.communication_label.configure(text='You loaded ' + str(self.answer2) + '. This session contains ' + str(len(self.im_list)) + ' images and ' + str(self.df.shape[1]) + ' clusters.')
                else:
                    self.communication_label.configure(text='You loaded ' + str(self.answer2) + '. This session contains ' + str(len(self.im_list)) + ' images. The features are calculated, but still need to be clustered.')
            else:
                self.communication_label.configure(text='You loaded ' + str(self.answer2) + '. This session contains ' + str(len(self.im_list)) + ' images. The features still need to be calculated.')
        else:
            self.communication_label.configure(text='You loaded ' + str(self.answer2) + '. This session is still fresh. Select an image folder on the left to begin.')
        
        
        
    #function to export (make a copy of) all the images in all the buckets to folders with the buckets' names, in a location specified by the user.        
    def export_buckets(self):
        self.catList = self.categories.get(0,END)        
        self.answer = filedialog.askdirectory()   #this will make the file directory a string
        for p in range(0,len(self.catList)):
            bucket = self.theBuckets[self.catList[p]]
            if len(bucket) > 0:
                bucket_dir = self.answer + '/' + self.catList[p]
                if not os.path.exists(bucket_dir):
                        os.makedirs(bucket_dir)
                for q in range(0,len(bucket)):
                    source_ = self.im_list[int(bucket[q])]
                    destination_ = bucket_dir + "/" + os.path.basename(source_)
                    if os.path.isfile(destination_):
                        shutil.copy2(source_, destination_ + id_generator())
                    else:
                        shutil.copy2(source_, destination_)
################################
################################
   
################################
###############################
            
    #function to select the image folder that the user wants to analyze
    def select_foldere(self):
        self.selected_folder = filedialog.askdirectory()   #this will make the file directory a string
        if len(self.selected_folder) > 0:
            self.im_list = glob.glob(self.selected_folder + '/**/*.jpg', recursive=True)
            self.im_list.extend(glob.glob(self.selected_folder + '/**/*.JPG', recursive=True))
            self.im_list.extend(glob.glob(self.selected_folder + '/**/*.gif', recursive=True))
            self.im_list.extend(glob.glob(self.selected_folder + '/**/*.GIF', recursive=True))
            self.im_list.extend(glob.glob(self.selected_folder + '/**/*.png', recursive=True))
            self.im_list.extend(glob.glob(self.selected_folder + '/**/*.PNG', recursive=True))
            self.im_list.extend(glob.glob(self.selected_folder + '/**/*.tif', recursive=True))
            self.im_list.extend(glob.glob(self.selected_folder + '/**/*.TIF', recursive=True))
            self.im_list.extend(glob.glob(self.selected_folder + '/**/*.bmp', recursive=True))
            self.im_list.extend(glob.glob(self.selected_folder + '/**/*.BMP', recursive=True))
            self.im_list = list(set(self.im_list))
            self.communication_label.configure(text='Continue by pressing the Calculate image features button')
            self.sel_folder.set('found ' + str(len(self.im_list)) + ' images in ' + self.selected_folder)

    #function to extract features from a neural network as defined below. New networks can be added if needed.
    
    
    def calculate_features(self):
        
        def feature_extraction_batch(neural_net, content): #content is list with image locations
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            
            
            #I commented out the above lines mostly for simplicity
            #The other models could also work with this code
            f_size = 2048
            model = models.resnet152(pretrained=True)
            #model = model.to(device)
            #This a step to allow multi-gpu, though actually later code will not yet work for multi-gpu
            #model = torch.nn.DataParallel(model).to(device)#.cuda()
            model = model.to(device)
            
            model.eval()
            
            #Define which layer to extract features from
            features_names = ['avgpool']
            #Features_blobs is simply a place to hold the features extracted from each pass
            #The fact that this is a list is due to a (future) possibility of multi-GPU support
            features_blobs = []
            
            #"hook_fatures" is the function that will run on each pass
            def hook_feature(module, input, output):
                features_blobs.append(np.squeeze(output.data.cpu().numpy()))
            #"register_forward_hook" tells pytorch what function to run on each pass
            for name in features_names:
                model._modules.get(name).register_forward_hook(hook_feature)
                #model.module._modules.get(name).register_forward_hook(hook_feature)
            
            
            
            mean=[0.485, 0.456, 0.406]
            std=[0.229, 0.224, 0.225]
        
            transform = transforms.Compose([
                        transforms.Resize((224,224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std)])
            
            
            class Dataset(data.Dataset):
                def __init__(self,imglist,transform=None):
        
                    if len(imglist) == 0:
                        raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                           "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        
                    self.imgs = imglist
                    self.transform = transform
        
                def __getitem__(self, index):
                    path = self.imgs[index]
                    target = None
                    img = Image.open(path).convert('RGB')
                    if self.transform is not None:
                        img = self.transform(img)
                    return img, path
        
                def __len__(self):
                    return len(self.imgs)
            #This creates a Pytorch dataset - a collection of images and the transform
            dataset = Dataset(content, transform) 
            
            #The loader provides details of how to load the dataset into the model
            #Well, more it is technical details of how to prepare the data for the model
            loader = data.DataLoader(
                dataset,
                batch_size=batch_size, #1024 images at a time
                num_workers=0,
                shuffle=False)
            
            #Let's make the return dataset
            #This is of size (n_images,vector size)
            return_shape = (len(content),f_size)
            
            #I found that in practise float16 is sufficient and saves memory, that is the 'dtype'
            return_array = np.zeros(return_shape)#,dtype='float16') 
            
            #Now let's iterate through the dataset
            #Bear in mind not to try this on multi-GPU setups
            num_batches = len(dataset) / batch_size
            for batch_idx, (input, paths) in enumerate(loader):
                if (batch_idx%100==0):
                    print(batch_idx, " / ", num_batches)
                    #print(datetime.datetime.now())
                del features_blobs[:]
                input = input.cuda()
                #input_var = Variable(input, volatile=True)
                input_var = torch.tensor(input)
                logit = model.forward(input_var)
                #Now we have run the forward, "hook_feature" should have run automatically
                #Placing the results of avgpool inside "features_blobs"
                
                #Now we set the returnarray data points to that found in feat
                start_idx = batch_idx*batch_size
                end_idx = min((batch_idx+1)*batch_size, len(dataset))
                
                return_array[start_idx:end_idx,:] = features_blobs[0]#astype('float16')
            return return_array
        


        def feature_extraction(neural_net, im_list):
            f = im_list
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            if neural_net == 'inception_v3':
                f_size = 2048
                model = models.inception_v3(pretrained='imagenet')
                layer = model._modules.get('Mixed_7c')
            elif neural_net == 'resnet152': #2084
                f_size = 2048
                model = models.resnet152(pretrained=True)
                layer = model._modules.get('avgpool')
            elif neural_net == 'resnet18': #512
                f_size =512
                model = models.resnet18(pretrained=True)
                layer = model._modules.get('avgpool')
            elif neural_net == 'vgg16': #4096
                f_size = 4096
                model = models.vgg16(pretrained=True)
                model.classifier = model.classifier[:-1]
                layer   = 'nothing'
            elif neural_net == 'vgg19': #4096
                f_size =    4096
                model = models.vgg19(pretrained=True)
                model.classifier = model.classifier[:-1]
                layer = 'nothing'
            elif neural_net == 'densenet161': #2208
                f_size =2208
                model = models.densenet161(pretrained=True)	
                model = model.features
            elif neural_net == 'squeezenet1_0': #512
                f_size = 1000
                model = models.squeezenet1_0(pretrained=True)
                #model = model.features
                layer = 'nothing'
            elif neural_net == 'alexnet':    
                f_size = 4096
                model = models.alexnet(pretrained=True)
                model.classifier = model.classifier[:-1]
                layer = 'nothing'
            model.eval()
            model = model.to(device)
            mean=[0.485, 0.456, 0.406]
            std=[0.229, 0.224, 0.225]
            if neural_net == 'inception_v3':
                transform = transforms.Compose([
                            transforms.Resize((299,299)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std)
                        ])
            else:
                transform = transforms.Compose([
                            transforms.Resize((224,224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std)
                        ])
            def get_vector(image_name,f_size, layer, transform, model, neural_net):
                try:
                    img = Image.open(image_name)                
                except OSError:
                    #if an image fails to open, a black replacement image is created...
                    img = Image.new('RGB',(100,100))
                
                if img.mode == 'RGB':
                    try:
                        t_img = transform(img).unsqueeze(0)
                    except OSError:
                        t_img = transform(img).unsqueeze(0)
                    t_img = t_img.to(device)
                    if neural_net == 'alexnet' or neural_net =='vgg19' or neural_net =='vgg16' or neural_net =='alexnet' or neural_net =='squeezenet1_0':
                        torch.cuda.empty_cache()
                        my_embeddingz = model(t_img)
                        my_embeddingz = my_embeddingz.cpu()
                        my_embeddingz = np.ndarray.flatten(my_embeddingz.data.numpy())
                    elif neural_net == 'densenet161':
                        featuresY = model(t_img)
                        my_embeddingz = F.relu(featuresY,inplace= True)
                        my_embeddingz = F.avg_pool2d(my_embeddingz, kernel_size=7, stride=1).view(featuresY.size(0), -1)
                        my_embeddingz = my_embeddingz.cpu()
                        my_embeddingz = np.ndarray.flatten(my_embeddingz.data.numpy())
                    elif neural_net == 'inception_v3':
                        my_embeddingz = torch.zeros((1,f_size,8,8))
                        my_embeddingz = my_embeddingz.to(device)
                            # 4. Define a function that will copy the output of a layer
                        def copy_data(m, i, o):
                            my_embeddingz.copy_(o.data)
                            # 5. Attach that function to our selected layer
                        h = layer.register_forward_hook(copy_data)
                            # 6. Run the model on our transformed image
                        model(t_img)
                        #    # 7. Detach our copy function from the layer
                        h.remove()
                        my_embeddingz = F.avg_pool2d(my_embeddingz, kernel_size=8)
                        my_embeddingz = my_embeddingz.view(my_embeddingz.size(0), -1)
                        my_embeddingz = my_embeddingz.cpu()
                        my_embeddingz = np.ndarray.flatten(my_embeddingz.data.numpy())
                    else:
                        my_embeddingz = torch.zeros((1,f_size,1,1))
                        my_embeddingz = my_embeddingz.to(device)
                            # 4. Define a function that will copy the output of a layer
                        def copy_data(m, i, o):
                            my_embeddingz.copy_(o.data)
                            # 5. Attach that function to our selected layer
                        h = layer.register_forward_hook(copy_data)
                            # 6. Run the model on our transformed image
                        model(t_img)
                        #    # 7. Detach our copy function from the layer
                        h.remove()
                        my_embeddingz = my_embeddingz.cpu()
                        my_embeddingz = np.ndarray.flatten(my_embeddingz.data.numpy())
                else:
                    my_embeddingz = np.zeros((f_size,))
                return my_embeddingz
            my_embed = []
#            self.progress["value"] = 0
#            self.progress["maximum"] = len(f)-1
            for i in range(0,len(f)):
                my_embed.append(get_vector(f[i],f_size,layer,transform,model,neural_net))
                print ("\r Extracting image {} out of {} images ".format(i+1,len(f)), end="")
#                if i%10 == 0:
                self.communication_label.configure(text='Processing image '+ str(i) + ' of ' + str(len(f)))
                #                    self.communication_label['background'] = '#99CCFF'
                self.communication_label.update()
#                self.communication_label.update()
            print("Finished extracting")
            features = np.asmatrix(my_embed)
            return features
        self.features = feature_extraction(self.neuralnet,self.im_list)
        self.communication_label.configure(text = 'Calculated the features. You can now start clustering the images by pressing Cluster images. You may also want to save now that the features are calculated')
        self.communication_label['background'] = '#FFFFFF'

        
        
        # function to calculate the correlations between all images and place them in a numpy matrix.
        def create_matrix_fast(features):
            cm = np.corrcoef(features)
            for i in range(0,features.shape[0]):
                cm[i,i]=None
            return cm
            
        def create_matrix(features,distance_metric):
            cm = []
            if distance_metric == 'correlation':
                features = np.squeeze(np.expand_dims(features,0))
                features_t = np.transpose(features)
                cm = []
                sumX = sum(features_t)
                sumsquareX = sum(features_t**2)
                for i in range(0,features.shape[0]):
                    feat0 = features[i]
                    sumXY = np.dot(feat0,features_t)
                    r = features.shape[1]*sumXY - sumX[i]*sumX 
                    s = ((features.shape[1] * sumsquareX) - sumX**2)
                    t = 1./((s[i]*s)**0.5)
                    u = r * t
                    cm.append(u)
                cm = np.asmatrix(cm)
                for i in range(0,features.shape[0]):
                    cm[i,i]=None
            elif distance_metric == 'l2':
                cm= []
                features = np.squeeze(np.expand_dims(features,0))
                #features = preprocessing.normalize(features, norm='l2')
                features_t = np.transpose(features)
                for i in range(0,features.shape[0]):
                    feat0 = features[i]
                    featuresX = features - feat0
                    featuresX = featuresX**2
                    sumIT = np.sum(featuresX,1)
                    sumIT = sumIT ** 0.5
                    cm.append(sumIT)
                cm = np.asmatrix(cm)
                cm = 1-cm
                for i in range(0,features.shape[0]):
                    cm[i,i]=None
        
            elif distance_metric == 'l1':
                cm= []
                features = np.squeeze(np.expand_dims(features,0))
                features_t = np.transpose(features)
                for i in range(0,features.shape[0]):
                    feat0 = features[i]
                    featuresX = features - feat0
                    featuresX = featuresX**2
                    sumIT = np.sum(featuresX,1)
                    cm.append(sumIT)
                cm = np.asmatrix(cm)
                cm = 1-cm            
                for i in range(0,features.shape[0]):
                    cm[i,i]=None
            elif distance_metric == 'euclidean':
                cm= []
                features = np.squeeze(np.expand_dims(features,0))
                features_t = np.transpose(features)
                for i in range(0,features.shape[0]):
                    feat0 = features[i]
                    featuresX = features - feat0
                    featuresX = featuresX**2
                    sumIT = np.sum(featuresX,1)
                    sumIT = sumIT / np.max(sumIT)
                    cm.append(sumIT)
                    print(i)
                cm = np.asmatrix(cm)
                cm = 1-cm            
                for i in range(0,features.shape[0]):
                    cm[i,i]=None
        
            elif distance_metric == 'cosine':
                cm = []
                features = np.squeeze(np.expand_dims(features,0))
                features_t = np.transpose(features)
                rootsumsquareX = sum(features_t**2)**0.5
                for i in range(0,features.shape[0]):
                    feat0 = features[i]
                    sXY = np.dot(feat0,features_t).transpose()
                    t_fea = sXY/(rootsumsquareX*rootsumsquareX[i])
                    cm.append(t_fea)
                cm = np.asarray(cm).astype('float32')
                for i in range(0,features.shape[0]):
                    cm[i,i]=None
        
            return cm
        #self.cm =create_matrix_fast(self.features,'correlation')
        self.cm =create_matrix_fast(self.features)
        
        
        
    #function to cluster all images based on the correlation score, as specified in []
    def cluster_images(self):
        self.communication_label.configure(text='Calculating the clusters. This may take several minutes, depending on the number of images.')
        self.communication_label['background'] = '#99CCFF'
        self.communication_label.update()
        try:
            self.threshold = float(self.threshold_entry.get())
        except ValueError:
            self.threshold = 0.5
        if self.threshold < 0:
            self.threshold = 0.5
        if self.threshold > 1:
            self.threshold = 0.5
        def clustering(cm,threshold):
            #threshold = 0.2
            
            cluster_ind = []
            weight2 = 1
            cm = np.nan_to_num(cm)
            m = len(cm)
            #cm_bool = copy.deepcopy(cm)
            #cm_bool = np.nan_to_num(cm_bool)
            cm_bool = np.zeros((m,m))
            cm_bool[cm >= threshold] = 1
            cm_bool[cm < threshold] = 0
            tt = 0
            while np.sum(cm) > m**2*-100:
                if tt > 0 and threshold > 0.1:
                    threshold = threshold-0.1
                    #cm_bool = np.zeros((len(cm),len(cm)))
                    #cm_bool = np.nan_to_num(cm_bool)
                    cm_bool[cm >= threshold] = 1
                    cm_bool[cm < threshold] = 0
                elif tt > 0 and threshold <= 0.1:
                    #cm_bool = copy.deepcopy(cm)
                    #cm_bool = np.nan_to_num(cm_bool)
                    cm_bool[cm > -100] = 1
                    cm_bool[cm == -100] = 0
                    cm_sum = np.sum(cm_bool,1)
                    remains = np.arange(0,m)
                    final_column = remains[cm_sum > 0]
                    cluster_ind.append(final_column)
                    cm[final_column,:] = -100
                    cm[:,final_column] = -100
                    break
                cm_sum = np.sum(cm_bool,1)
                tt = 1
                new_cluster = []
                #while np.sum(cm_bool) > 0:
                while np.sum(cm_sum) - len(new_cluster)  > 0:
                    tg = 0
                    for g in range(0,len(cluster_ind)):
                        tg = tg + len(cluster_ind[g])
                    #cm_sum = np.sum(cm_bool,1)
                    cm_most = np.argmax(cm_sum)
                    column1 = cm[:,cm_most]
                    new_cluster = []
                    column_max = np.argmax(column1)
                    if column1[column_max] > threshold:
                        column2 = cm[:,column_max]
                        weight1 = 1        
                        new_column = float(weight1) / (weight1+weight2) * column1 + float(weight2)/(weight1+weight2)*column2
                        cm_bool[:,cm_most] = 0
                        cm_bool[cm_most,:] = 0
                        cm_bool[column_max,:] = 0
                        cm_bool[:,column_max] = 0
                        new_cluster.append(cm_most)
                        new_cluster.append(column_max)
                        new_column[new_cluster] = 0
                        while bn.nanmax(new_column) > threshold:
                            weight1 = weight1 + 1
                            column1 = new_column
                            column_max = bn.nanargmax(column1)
                            column2 = np.nan_to_num(cm[:,column_max])
                            new_column = float(weight1) / (weight1+weight2) * column1 + float(weight2)/(weight1+weight2)*column2
                            cm_bool[column_max,:] = 0
                            cm_bool[:,column_max] = 0
                            new_cluster.append(column_max)
                            new_column[new_cluster] = 0
                    else:
                        break
                    cluster_ind.append(new_cluster)
                    cm_sum[new_cluster] = 0
                    cm[new_cluster,:] = -100
                    cm[:,new_cluster] = -100
            xlength = []
            for q in range(0,len(cluster_ind)):
                xlength.append(len(cluster_ind[q]))
            cluster_indX = np.zeros((max(xlength),len(cluster_ind)))-1        
            for r in range(0,len(cluster_ind)):
                for s in range(0,len(cluster_ind[r])):
                    cluster_indX[s,r] = int(cluster_ind[r][s])
            cluster_indX = cluster_indX.astype(int)
        
                
                
            return cluster_indX

#        def clustering(cm,threshold):
#            hckz = 0 
#            weight2 = 1
#            cluster_ind =[]
#            while bn.allnan(cm) == False:
#                cmmax = bn.nanmax(cm,1)
#                print(hckz)
#                xr = bn.nanmax(cmmax)
#                if xr<threshold:
#                    threshold = xr - 0.1
#                while xr >= threshold:
#                   try: 
#                    hckz = hckz + 2 #just a way to keep track of progress, but it never worked
#                    weight1 = 1
#                    cmmax = bn.nanmax(cm,1) # this is the max that seems to be the bottleneck 
#                    ind_max = []
#                    ind_max.append(bn.nanargmax(cmmax))
#                    ind_max.append(bn.nanargmax(cm[ind_max[0]]))
#                    new_cluster = []
#                    the_max = cm[ind_max[1],ind_max[0]]
#                    column1 = cm[:,ind_max[0]]
#                    column2 = cm[:,ind_max[1]]
#                    new_colx = float(weight1) / (weight1+weight2) * column1
#                    new_coly = float(weight2)/(weight1+weight2)*column2
#                    new_column = float(weight1) / (weight1+weight2) * column1 + float(weight2)/(weight1+weight2)*column2
#                    cm[:,ind_max[0]] = None#new_column
#                    cm[ind_max[0],:] = None
#                    cm[ind_max[1],:] = None
#                    cm[:,ind_max[1]] = None
#                    new_cluster.append(ind_max[0])
#                    new_cluster.append(ind_max[1])
#                    while np.nanmax(new_column) >= threshold:
#                        hckz = hckz +1
#                        column1 = new_column
#                        next_ind_max = bn.nanargmax(column1,0)
#                        nextmax = cm[next_ind_max,ind_max[0]]
#                        column2 = cm[:,next_ind_max]
#                        weight1 = weight1 + 1
#                        new_column = float(weight1) / (weight1+weight2) * column1 + float(weight2)/(weight1+weight2)*column2
#                        cm[next_ind_max,:] = None
#                        cm[:,next_ind_max] = None
#                        new_cluster.append(next_ind_max)
#                        cm[:,ind_max[0]] = None#new_column
#                    cluster_ind.append(new_cluster)
#                    cmmax = bn.nanmax(cm,1)
#                    xr = bn.nanmax(cmmax)
#                   except ValueError:
#                       cmmax = bn.nanmax(cm,1)
#                       ind_max = []
#                       ind_max.append(bn.nanargmax(cmmax))
#                       ind_max.append(bn.nanargmax(cm[ind_max[0]]))
#                       cluster_ind.append(bn.nanmax(ind_max))
#            xlength = []
#            for q in range(0,len(cluster_ind)):
#                xlength.append(len(cluster_ind[q]))
#            cluster_indX = np.zeros((max(xlength),len(cluster_ind)))-1        
#            for r in range(0,len(cluster_ind)):
#                for s in range(0,len(cluster_ind[r])):
#                    cluster_indX[s,r] = int(cluster_ind[r][s])
#            cluster_indX = cluster_indX.astype(int)
#            return cluster_indX
        self.df = clustering(copy.deepcopy(self.cm),self.threshold)
        self.df = pd.DataFrame.from_records(self.df)
        self.communication_label.configure(text='Calculated the clusters. You can now start browsing the clusters by Next cluster. You could also preload the images for a smoother browsing experience. Preloading takes a moment initially and uses a bit more system memory.')
        self.communication_label['background'] = '#FFFFFF'
        num_c = 0
        self.num_c = 0

    #function to calculate the average features of a cluster. This way, the most representative image of a cluster can be found. Doing so, an overview of all clusters using the representative image can be generated.
    def calculate_avg_vector(self):
        self.communication_label.configure(text='Calculating the vector. Please wait a moment.')
        self.communication_label['background'] = '#99CCFF'
        self.communication_label.update_idletasks()
        df = self.df + 1
        nonzeros = df.astype(bool).sum(axis=0)
        df = None
        avg_feat_vec = []
        self.cluster_to_vector = []
        self.represent=[]
        for i in range(0,len(self.df.columns)):
            avg_feat = np.zeros([nonzeros[i],np.size(self.features,1)])
            for j in range(0,nonzeros[i]):
                avg_feat[j,0:np.size(self.features,1)] = self.features[self.df[i][j]]
            avg_feat_vec.append(np.mean(avg_feat,axis=0))
            calc_corr = []
            for k in range(0,nonzeros[i]):
                cc = np.corrcoef(avg_feat_vec[i],avg_feat[k])
                calc_corr.append(cc[0][1])
            self.represent.append(self.df[i][np.argmax(calc_corr)])
            self.cluster_to_vector.append(calc_corr)
        self.communication_label.configure(text='The vector has been calculated. You can now press Show overview to see a representative image of each cluster. You can then select a cluster and view it by pressing Show selected cluster.')
        self.communication_label['background'] = '#99CCFF'

    #function to show the overview of representative images for each cluster.    
    def show_overview(self):
        try:
            del self.ind_for_overview
            del self.xsorted
            del self.xsortedimtags
        except AttributeError:
            pass
        self.bucketDisp = 0
        self.c.delete("all")
        self.im_numX = []
        self.imagex = []
        self.display_images(self.represent)

    #Function that displays the cluster selected by the user from the overview of representative images
    def show_selected_cluster(self):
        im_num = self.selected_images
        
        if im_num:
            try:            
                im_num = self.ind_for_overview[im_num]
            except AttributeError:
                pass
            self.c.delete("all")
            #self.num_clus = im_num
            self.c.xview_moveto(self.origX)  #####
            self.c.yview_moveto(self.origY) ######
            cluster = []
            for tt in range(len(im_num)):
                print(im_num[tt])
                cluster.append(self.df[im_num[tt]])
            cluster = np.asarray(cluster)
            cluster = cluster[cluster > -1]
            num_im =int(self.e1.get())
            self.imagex = []
        if num_im > len(cluster):
            num_im = len(cluster)
        self.display_images(cluster)
    #function to create and display a TSNE graph, where the user can select which points to view as images
    def create_tsne(self):
        self.tsne_squares = None
        #function to allow the user to select points in the TSNE graph by drawing a square
        def tsne_click(event):
            self.tsneclick = self.tsneclick + 1
            if self.tsneclick == 2:
                self.evex_tsne2 = self.canvas_tsne.canvasx(event.x)
                self.evey_tsne2 = self.canvas_tsne.canvasy(event.y)
                self.tsneclick = 0
                if self.tsne_squares is not None:
                    self.canvas_tsne.delete(self.tsne_squares)
                self.tsne_squares = self.canvas_tsne.create_rectangle(self.evex_tsne1, self.evey_tsne1, self.evex_tsne2, self.evey_tsne2)                    
#                print(self.evex_tsne1)
#                print(self.evex_tsne2)
#                print(self.evey_tsne1)
#                print(self.evey_tsne2)
                xmin = (self.evex_tsne1-1)/(self.screen_width/2)
                xmax = (self.evex_tsne2-1)/(self.screen_width/2) 
                ymin = 1 - ((self.evey_tsne1-1)/(self.screen_height))
                ymax = 1 - ((self.evey_tsne2-1)/(self.screen_height))
#                print(xmin)
#                print(xmax)
#                print(ymin)
#                print(ymax)
#                print('hh')
#                print(self.screen_height)
#                print(self.screen_width)
                #sloppy implementation of offsetting the user's selection...:
#                xmin = (xmin -0.065) * (1/(0.94-0.065))
#                xmax = (xmax -0.065) * (1/(0.94-0.065))
#                ymin = (ymin - 0.115) * (1/(0.94-0.115))
#                ymax = (ymax - 0.115) * (1/(0.94-0.115))
                X_images = np.where((self.X_embed[:,0] > np.min((xmin,xmax))) & (self.X_embed[:,0] < np.max((xmin,xmax))) & (self.X_embed[:,1] > np.min((ymin,ymax))) & (self.X_embed[:,1] < np.max((ymin,ymax))))


#                X_images = self.X_embed[self.X_embed[:,0] < np.max((xmin,xmax))]
#                X_images = X_images[X_images[:,0] > np.min((xmin,xmax))]
#                X_images = X_images[X_images[:,1] < np.max((ymin,ymax))]
#                X_images = X_images[X_images[:,1] > np.min((ymin,ymax))]

                
                self.display_images(X_images[0])
            else:
                self.evex_tsne1 = self.canvas_tsne.canvasx(event.x)
                self.evey_tsne1 = self.canvas_tsne.canvasy(event.y)
            
        if len(self.X_embed) < 1: #get TSNE embedding and normalize it.
            self.X_embed = TSNE().fit_transform(self.features)
            self.X_embed[:,0] = self.X_embed[:,0] + abs(np.min(self.X_embed[:,0]))
            self.X_embed[:,1] = self.X_embed[:,1] + abs(np.min(self.X_embed[:,1]))
            self.X_embed[:,0] = self.X_embed[:,0]/np.max(self.X_embed[:,0])
            self.X_embed[:,1] = self.X_embed[:,1]/np.max(self.X_embed[:,1])

        try:
            self.canvas.delete("all")
        except AttributeError:
            pass

        plt.ioff()
        mpl.rcParams['savefig.pad_inches'] = 0
#        self.X_embed = [0,0.1,0.2,0.3,0.4,0.98,1]
#        self.X_embed = np.vstack((self.X_embed,self.X_embed)).transpose()
        self.canvas_tsne = Canvas(self.newWindow,bg='#555544',bd=0,highlightthickness=0, width = self.screen_width/2+1, height =self.screen_height+1) #canvas size
        self.canvas_tsne.place(x = self.screen_width/2, y=100)
        self.tsne = plt.figure(figsize=(self.screen_width/100/2,self.screen_height/100), dpi=100,facecolor='#555555' )

        self.tsne_plot = self.tsne.add_axes([0,0,1,1], facecolor='#FFFFFF',frameon=False)
#        in_bucket = []
#        not_in_bucket = []
#        for jt in range(0,len(self.im_list)):
#            if jt in [xx for vv in self.theBuckets.values() for xx in vv]:
#                in_bucket.append(jt)
#            else:
#                not_in_bucket.append(jt)
        t = list(chain.from_iterable(self.theBuckets.values()))
        xx = list(np.arange(0,len(self.im_list)))
        not_in_bucket = list(sorted(set(np.arange(0,len(self.im_list))) - set(t), key=xx.index))
        in_bucket = t
            
        self.tsne_plot.scatter(self.X_embed[not_in_bucket,0],self.X_embed[ not_in_bucket,1],c='black')
        self.tsne_plot.scatter(self.X_embed[in_bucket,0],self.X_embed[in_bucket,1],c='#00EEEE')
        try:
            elected_buckets = self.filter_in

            color_bucket = []
            for i in range(0,len(elected_buckets)):
                color_bucket.append(self.theBuckets[self.categories.get(elected_buckets[i])])

            color_bucket = list(chain.from_iterable(color_bucket))
            if len(color_bucket) > 0:
                self.tsne_plot.scatter(self.X_embed[color_bucket,0],self.X_embed[color_bucket,1],c='red')
                
        except AttributeError:
            pass

        
        #self.tsne_plot.set_xlabel('TSNE graph, make a selection!',color='#FFFFFF')
        self.tsne_plot.tick_params(axis='x', colors='#FFFFFF')
        self.tsne_plot.tick_params(axis='y', colors='#FFFFFF')
        self.tsne_plot.spines['bottom'].set_color('#FFFFFF')
        self.tsne_plot.spines['top'].set_color('#FFFFFF') 
        self.tsne_plot.spines['right'].set_color('#FFFFFF')
        self.tsne_plot.spines['left'].set_color('#FFFFFF')
        #self.tsne_plot.axes.get_xaxis().set_visible(False)
        self.tsne_plot.axes.get_yaxis().set_visible(False)
        self.tsne_plot.axes.get_xaxis().set_ticks([])
        self.tsne_plot.axis([0,1.001,0,1.001])
        #self.canvas_tsne = Canvas(self.newWindow,bg='#666666',bd=0, width = self.screen_width/2, height =self.screen_height/2-200) #canvas size
        self.buftsne = io.BytesIO()

        self.tsne.savefig(self.buftsne, format='png', bbox_inches='tight', dpi=100,facecolor='#FFFFFF')

        load_tsne = Image.open(self.buftsne)
#        load_sankey = load_sankey.resize((self.screen_width-600,self.screen_height+300))
        render_tsne = ImageTk.PhotoImage(load_tsne)
        my_img = ttk.Label(self,background='#555555')
        my_img.image = render_tsne
        #image_.append(my_img)
#        self.canvas_tsne.create_image(int((self.screen_width-500)/2),int((self.screen_height+300)/2),image = render_tsne)
        self.canvas_tsne.create_image(load_tsne.size[0]/2,load_tsne.size[1]/2,image = render_tsne)

        
        #self.f.tight_layout()
#        self.canvas_tsne2 = FigureCanvasTkAgg(self.tsne, master=self.canvas_tsne)
##        self.gcanvas.show()
#        self.canvas_tsne2.get_tk_widget().pack(side=BOTTOM, fill=BOTH, expand=1)
        self.canvas_tsne.bind("<Button-1>", tsne_click)

        
        
        
        
        #self.canvas.create_image(int((self.screen_width-500)/2),int((self.screen_height+300)/2),image = render_sankey)
    
    def create_sankey(self):
        def sankey2(left, right, leftWeight=None, rightWeight=None, colorDict=None,
                   leftLabels=None, rightLabels=None, aspect=4, rightColor=False,
                   fontsize=8, figureName=None, closePlot=False):
            '''
            Make Sankey Diagram showing flow from left-->right
            Inputs:
                left = NumPy array of object labels on the left of the diagram
                right = NumPy array of corresponding labels on the right of the diagram
                    len(right) == len(left)
                leftWeight = NumPy array of weights for each strip starting from the
                    left of the diagram, if not specified 1 is assigned
                rightWeight = NumPy array of weights for each strip starting from the
                    right of the diagram, if not specified the corresponding leftWeight
                    is assigned
                colorDict = Dictionary of colors to use for each label
                    {'label':'color'}
                leftLabels = order of the left labels in the diagram
                rightLabels = order of the right labels in the diagram
                aspect = vertical extent of the diagram in units of horizontal extent
                rightColor = If true, each strip in the diagram will be be colored
                            according to its left label
            Ouput:
                None
            '''
            if leftWeight is None:
                leftWeight = []
            if rightWeight is None:
                rightWeight = []
            if leftLabels is None:
                leftLabels = []
            if rightLabels is None:
                rightLabels = []
            # Check weights
            if len(leftWeight) == 0:
                leftWeight = np.ones(len(left))
        
            if len(rightWeight) == 0:
                rightWeight = leftWeight
            
            figuur = plt.Figure()
            a = figuur.add_subplot(111)

            plt.rc('text', usetex=False)
            plt.rc('font', family='serif')
        
            # Create Dataframe
            if isinstance(left, pd.Series):
                left = left.reset_index(drop=True)
            if isinstance(right, pd.Series):
                right = right.reset_index(drop=True)
            dataFrame = pd.DataFrame({'left': left, 'right': right, 'leftWeight': leftWeight,
                                      'rightWeight': rightWeight}, index=range(len(left)))
        
            if len(dataFrame[(dataFrame.left.isnull()) | (dataFrame.right.isnull())]):
                raise NullsInFrame('Sankey graph does not support null values.')
        
            # Identify all labels that appear 'left' or 'right'
            allLabels = pd.Series(np.r_[dataFrame.left.unique(), dataFrame.right.unique()]).unique()
        
            # Identify left labels
            if len(leftLabels) == 0:
                leftLabels = pd.Series(dataFrame.left.unique()).unique()
            else:
                check_data_matches_labels(leftLabels, dataFrame['left'], 'left')
        
            # Identify right labels
            if len(rightLabels) == 0:
                rightLabels = pd.Series(dataFrame.right.unique()).unique()
            else:
                check_data_matches_labels(leftLabels, dataFrame['right'], 'right')
            # If no colorDict given, make one
            if colorDict is None:
                colorDict = {}
                palette = "hls"
                colorPalette = sns.color_palette(palette, len(allLabels))
                for i, label in enumerate(allLabels):
                    colorDict[label] = colorPalette[i]
            else:
                missing = [label for label in allLabels if label not in colorDict.keys()]
                if missing:
                    msg = "The colorDict parameter is missing values for the following labels : "
                    msg += '{}'.format(', '.join(missing))
                    raise ValueError(msg)
        
            # Determine widths of individual strips
            ns_l = defaultdict()
            ns_r = defaultdict()
            for leftLabel in leftLabels:
                leftDict = {}
                rightDict = {}
                for rightLabel in rightLabels:
                    leftDict[rightLabel] = dataFrame[(dataFrame.left == leftLabel) & (dataFrame.right == rightLabel)].leftWeight.sum()
                    rightDict[rightLabel] = dataFrame[(dataFrame.left == leftLabel) & (dataFrame.right == rightLabel)].rightWeight.sum()
                ns_l[leftLabel] = leftDict
                ns_r[leftLabel] = rightDict
        
            # Determine positions of left label patches and total widths
            leftWidths = defaultdict()
            for i, leftLabel in enumerate(leftLabels):
                myD = {}
                myD['left'] = dataFrame[dataFrame.left == leftLabel].leftWeight.sum()
                if i == 0:
                    myD['bottom'] = 0
                    myD['top'] = myD['left']
                else:
                    myD['bottom'] = leftWidths[leftLabels[i - 1]]['top'] + 0.02 * dataFrame.leftWeight.sum()
                    myD['top'] = myD['bottom'] + myD['left']
                    topEdge = myD['top']
                leftWidths[leftLabel] = myD
        
            # Determine positions of right label patches and total widths
            rightWidths = defaultdict()
            for i, rightLabel in enumerate(rightLabels):
                myD = {}
                myD['right'] = dataFrame[dataFrame.right == rightLabel].rightWeight.sum()
                if i == 0:
                    myD['bottom'] = 0
                    myD['top'] = myD['right']
                else:
                    myD['bottom'] = rightWidths[rightLabels[i - 1]]['top'] + 0.2 * dataFrame.rightWeight.sum()
                    myD['top'] = myD['bottom'] + myD['right']
                    topEdge = myD['top']
                rightWidths[rightLabel] = myD
        
            # Total vertical extent of diagram
            xMax = topEdge / aspect
        
            # Draw vertical bars on left and right of each  label's section & print label
            for leftLabel in leftLabels:
                plt.fill_between(
                    [-0.02 * xMax, 0],
                    2 * [leftWidths[leftLabel]['bottom']],
                    2 * [leftWidths[leftLabel]['bottom'] + leftWidths[leftLabel]['left']],
                    color=colorDict[leftLabel],
                    alpha=0.99
                )
                plt.text(
                    -0.05 * xMax,
                    leftWidths[leftLabel]['bottom'] + 0.5 * leftWidths[leftLabel]['left'],
                    leftLabel,
                    {'ha': 'right', 'va': 'center'},
                    fontsize=fontsize,
                    color='#FFFFFF'
                )
            for rightLabel in rightLabels:
                plt.fill_between(
                    [xMax, 1.02 * xMax], 2 * [rightWidths[rightLabel]['bottom']],
                    2 * [rightWidths[rightLabel]['bottom'] + rightWidths[rightLabel]['right']],
                    color=colorDict[rightLabel],
                    alpha=0.99
                )
                plt.text(
                    1.05 * xMax,
                    rightWidths[rightLabel]['bottom'] + 0.5 * rightWidths[rightLabel]['right'],
                    rightLabel,
                    {'ha': 'left', 'va': 'center'},
                    fontsize=fontsize,
                    color='#FFFFFF'
                )
        
            # Plot strips
            for leftLabel in leftLabels:
                for rightLabel in rightLabels:
                    labelColor = leftLabel
                    if rightColor:
                        labelColor = rightLabel
                    if len(dataFrame[(dataFrame.left == leftLabel) & (dataFrame.right == rightLabel)]) > 0:
                        # Create array of y values for each strip, half at left value,
                        # half at right, convolve
                        ys_d = np.array(50 * [leftWidths[leftLabel]['bottom']] + 50 * [rightWidths[rightLabel]['bottom']])
                        ys_d = np.convolve(ys_d, 0.05 * np.ones(20), mode='valid')
                        ys_d = np.convolve(ys_d, 0.05 * np.ones(20), mode='valid')
                        ys_u = np.array(50 * [leftWidths[leftLabel]['bottom'] + ns_l[leftLabel][rightLabel]] + 50 * [rightWidths[rightLabel]['bottom'] + ns_r[leftLabel][rightLabel]])
                        ys_u = np.convolve(ys_u, 0.05 * np.ones(20), mode='valid')
                        ys_u = np.convolve(ys_u, 0.05 * np.ones(20), mode='valid')
        
                        # Update bottom edges at each label so next strip starts at the right place
                        leftWidths[leftLabel]['bottom'] += ns_l[leftLabel][rightLabel]
                        rightWidths[rightLabel]['bottom'] += ns_r[leftLabel][rightLabel]
                        plt.fill_between(
                            np.linspace(0, xMax, len(ys_d)), ys_d, ys_u, alpha=0.65,
                            color=colorDict[labelColor]
                        )
            plt.gca().axis('off')
            plt.gcf().set_size_inches(6, 6)
            self.bufsankey = io.BytesIO()
            if figureName != None:
#                plt.savefig("{}.png".format(figureName), bbox_inches='tight', dpi=150,facecolor='#555555')
                plt.savefig(self.bufsankey, format='png', bbox_inches='tight', dpi=150,facecolor='#555555')

            plt.close()

        #this uses the def sankey2 to create sankey
        def pd_for_sankey(thedict):
            mah_list = []       
            keyslist = []
            for key in thedict:
                mah_list.append(np.array(thedict[key]))
                keyslist.append(key)
            #themax = get_longest(mah_list)
            leftWeight = []
            rightWeight  = []
            bucket1 = []
            bucket2 = []
            for t in range(0,len(mah_list)):
                for u in range(len(mah_list)):
                    if t == u:
                        pass
                    else:
                        if len(mah_list[t]) == 0:
                            pass
                        elif len(list(set(mah_list[t]).intersection(mah_list[u]))) == 0:
                            pass
                        else:
                            leftWeight.append(len(mah_list[t]))
                            rightWeight.append(len(list(set(mah_list[t]).intersection(mah_list[u]))))
                            bucket1.append(keyslist[t])
                            bucket2.append(keyslist[u])
            for_pd = [bucket1,bucket2,rightWeight,leftWeight]
                
            the_df = pd.DataFrame.from_records(for_pd,index=None)
            the_df = pd.DataFrame.transpose(the_df)
            return the_df
        the_df = pd_for_sankey(self.theBuckets)
        sankey2(left=the_df[0], right=the_df[1], leftWeight=the_df[3], rightWeight=the_df[2], colorDict = None,aspect=20,fontsize=12,figureName='temp_for_sankey')
        self.frame = Frame(self.newWindow)
        self.frame.place(x=0,y=0)
        self.canvas = Canvas(self.newWindow,bg='#555555',bd=0, width =self.screen_width-500, height =self.screen_height+400) #canvas size
        self.canvas.place(x = 600, y=0)
#        load_sankey = Image.open('temp_for_sankey.png')
        load_sankey = Image.open(self.bufsankey)
        load_sankey = load_sankey.resize((self.screen_width-600,self.screen_height+300))
        render_sankey = ImageTk.PhotoImage(load_sankey)
        my_img = ttk.Label(self,background='#555555')
        my_img.image = render_sankey
        #image_.append(my_img)
        self.canvas.create_image(int((self.screen_width-500)/2),int((self.screen_height+300)/2),image = render_sankey)
    


    def client_exit(self):
        exit()

    
root = Tk()
root.geometry("1900x700")
root.configure(background='#555555')
app = Application(root)

root.mainloop()  
