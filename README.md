# IMEX
Incident Image Explorer

## Goal of Imex 

Incident Image Explorer (Imex) is an application for exploring large image collections in order to gain insight. Imex helps with efficiently structuring an image collection into user defined categories, by generating clusters of images based on the similarity of the images. This allows you to quickly browse through a semi-structured image collection. You can then use your expert knowledge to explore and search through the image collection, and structure the images as you want. 

## Quick start guide 

The Quick start guide will go over the basic features to get you started.

### The basics 

Click on “Select image folder” and browse to the folder containing your images. 

Click on “Calculate image features”. 

Click on “Cluster images”. The cluster threshold can be set (default 0.5) before clustering. Images of the first cluster will now be displayed. 

By default, the first 20 images of a cluster are shown. Adjust the number above the “Show images” button, and click the button to display more or fewer images. Click on “next cluster >” to view the next cluster. 

By default, two buckets have been created: “RelevantItems” and “Non-RelevantItems”. By selecting one of these buckets, and by then clicking on the “Add cluster to selected bucket(s)” button, the cluster will be added to the selected bucket. Alternatively, a selection of the displayed images can be made. Shift and Control keys on the keyboard can be used to select multiple images. The images can then be added to the selected bucket with the “Add selection to selected bucket(s)” button.  

To add a new bucket, add a name in the entry field above the “Add bucket >” button, and click the button. Shift and Control keys can be used to selected multiple buckets. 

To view the content of a bucket, selected the bucket and press the “Show bucket” button. 

Click on “Save session as” to save your current session. You can later load it through the “Load session” button. 

### Advanced features 

“Preload images”. As it can take a moment to display a large number of images, in case of a large cluster or large bucket, all images can be preloaded, making the browsing experience much faster and smoother. Please note that you may run out of memory, causing the application to crash, when your image collection is very large.  

“Expand current cluster”. The application will try to find additional images that are similar to the images from the cluster you are currently at. You can click it multiple times. To add newly found images to a bucket, you have to selected them and use the “Add selection to selected bucket(s)” button. 

“Export buckets”. With this button you can select a folder where a copy of all the images in the buckets will be placed, in subfolders with their respective bucket names.  

“Check this to hide Non-relevant images from displayed results”. Check this box to hide images that you have placed in the “Non-RelevantItems” bucket, so they do no longer show up in any of your results. 

“Check to hide images already in a bucket”. Check this box to hide images that you have already placed in one or more buckets. Especially useful in combination with the “Rank selected image” function.  

“Rank selected image”. By selecting one image, and clicking this button, Imex will rank all images in the image collection based on similarity to the selected image. The most similar image will be shown on top. Right-clicking an image will also activate this function.  

“Calculate overview”. For each cluster, the most representative image will be calculated. 

“Show overview”. For each cluster, the most representative image will be shown. This way you can quickly find relevant clusters. Select the clusters you want to view and press the “Show selected cluster” button, will show images in the cluster. 

“Focus image”. This button will show a larger version of the selected image. You can then draw a square by clicking on two points in the image. By then clicking on the “Query selection” button, you can search for similar images based only on the content in the square, rather than the whole image.  

“Set the image display size in pixels”. The entry box allows you to adjust the size of the images displayed (default is 100). The next time a set of images is displayed, it will be shown in the size you entered. 

Double clicking an image will open it in your default image browser.  

On the second window, two lists with your buckets are shown. By selecting multiple buckets on the left list, only images that are present in all selected buckets will be shown, after clicking the “Show filtered buckets” button. By selecting one or more buckets from the right list, only images not present in those buckets will be shown. 

“Create Sankey diagram”. Creates a Sankey diagram based on the images in your buckets. 
