from spire.pdf.common import *
from spire.pdf import *
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

doc = PdfDocument()

# Load a PDF document
pdf_path = ''
doc.LoadFromFile(pdf_path)

# Create a PdfImageHelper object
image_helper = PdfImageHelper()
# Iterate through the pages in the document
j = 0
for i in range(doc.Pages.Count):
    # Get the image information from the current page
    images_info = image_helper.GetImagesInfo(doc.Pages[i])
    # Get the images and save them as image files
    for image in images_info:
        if image.Bounds.Width>100 and image.Bounds.Height>100:
            if not os.path.exists('./Images_temp/'):
                os.makedirs('./Images_temp/')
            output_file = f"./Images_temp/image{j}.png"
            image.Image.Save(output_file)

            j += 1

doc.Close()
