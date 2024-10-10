import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tkinter as tk
import tkinter.simpledialog

from tkinter.messagebox import askyesno, showinfo
from tkinter.filedialog import askopenfilename
from PIL import Image


def remove_outliers(df, threshold=0.5, n_points=2, drop=1):
    """
    Removes outliers from a set of y-values based on distance deviation.
    
    Parameters:
        y: list or np.array - y-coordinates of the points
        threshold: float - deviation threshold for considering a point as an outlier
        n_points: int - number of initial points to ignore in outlier detection
    
    Returns:
        filtered_x: np.array - x coordinates with outliers removed
        filtered_y: np.array - y coordinates with outliers removed
    """
    
    df = df.tail(-drop).reset_index(drop=True)
    x = df.x
    y = df.y
    
    differences = np.array([y[i] - np.mean(y[i-n_points:i]) for i in range(n_points, len(y))])
    median_diff = np.median(differences[n_points:])
    std_diff = np.std(differences[n_points:])
    
    outlier_mask = np.abs(differences - median_diff) > (threshold * std_diff)
    mask = np.ones_like(y, dtype=bool)
    
    mask[n_points:][outlier_mask] = False
    mask[:n_points] = True
    
    filtered_y = y[mask]
    filtered_x = x[mask]
    
    return filtered_x, filtered_y
    

class GUI(tk.Tk):
    def __init__(self):
        super().__init__()

        # Setting up the App
        self.title("Image2Curve")
        self.resizable(False, False)
        self.HomePage = HomePage(self)


class HomePage(tk.Frame):
    def __init__(self, parent):
        super().__init__()

        self.parent = parent

        # open the image (in PNG or JPG)
        graph = askopenfilename(parent=self.parent, initialdir="./",
                                title='Choose the image',
                                filetypes=[('PNG', '*.png'), ('JPG', '*.jpg')])
        image = Image.open(graph)
        self.img = tk.PhotoImage(file=graph, master=self.parent)
        self.my_canvas = tk.Canvas(self.parent, width=image.size[0], height=image.size[1])
        self.my_canvas.pack()
        self.my_canvas.create_image(0, 0, image=self.img, anchor="nw")
    
        self.Xmin, self.Xmax = 0, 0
        self.Ymin, self.Ymax = 0, 0
        self.xmax_pixel, self.xmin_pixel = 0, 0
        self.ymax_pixel, self.ymin_pixel = 0, 0
        self.x0, self.x1 = 0, 0
        self.y0, self.y1 = 0, 0
        
        self.R, self.G, self.B = 0, 0, 0

        self.ranges()

    def ranges(self):
        # Defines XY ranges of the curve (don't need to be the same as in the image)
        self.Xmin = tk.simpledialog.askfloat("Xmin", "Left X value")
        self.Xmax = tk.simpledialog.askfloat("Xmax", "Right X value")
        self.Ymin = tk.simpledialog.askfloat("Ymin", "Lower Y value")
        self.Ymax = tk.simpledialog.askfloat("Ymax", "Upper Y value")
        self.define()

    def define(self):
        # Progressive clicking on important points (coordinations)
        def getOrigin(event):
            self.xmin_pixel = int(event.x)
            self.ymax_pixel = int(event.y)

            tk.messagebox.showinfo("Instructions", "Click on the X maximal tick")
            self.my_canvas.bind("<Button>", getXmax)

        def getXmax(event):
            self.xmax_pixel = int(event.x)
            tk.messagebox.showinfo("Instructions", "Click on the Y maximal tick")
            self.my_canvas.bind("<Button>", getYmin)

        def getYmin(event):
            self.ymin_pixel = int(event.y)
            self.legend()

        if not askyesno("Ranges defined",
                        f"X={self.Xmin,self.Xmax}, Y={self.Ymin, self.Ymax}"):
            self.ranges()

        showinfo("Instructions", "Click on the axes origin")
        self.my_canvas.bind("<Button>", getOrigin)

    def legend(self):
        # Extracting the corresponding colour (ignores legend if present)
        def colour(event):
            self.R, self.G, self.B = self.img.get(event.x, event.y)
            dialog = tk.Toplevel(self.parent)
            dialog.title("Chosen color")
            dialog.resizable(False, False)

            label = tk.Label(dialog, text="Is this the color you wanted?")
            square = tk.Canvas(dialog, width=50, height=50,
                               bg=f'#{self.R:02x}{self.G:02x}{self.B:02x}')

            yes_button = tk.Button(dialog, text="Yes",
                                   command=lambda: response(self, dialog, True))

            no_button = tk.Button(dialog, text="No",
                                  command=lambda: response(self, dialog, False))
            label.pack(pady=10)
            square.pack(pady=10)
            yes_button.pack(side="left", padx=20)
            no_button.pack(side="right", padx=20)

        def response(self, dialog, answer):
            if answer:
                dialog.destroy()
                self.extract()
            else:
                dialog.destroy()
                self.my_canvas.bind("<Button>", colour)

        def legend_up(event):
            self.x0 = event.x
            self.y0 = event.y
            self.my_canvas.bind("<Button>", legend_low)

        def legend_low(event):
            self.x1 = event.x
            self.y1 = event.y
            showinfo("Curve colour definition", "Click on the curve you want to extract")
            self.my_canvas.bind("<Button>", colour)

        if askyesno("Legend", "Is there a legend?", default='no'):
            showinfo("Legend removal",
                     "Click on the upper left then the lower right corner")
            self.my_canvas.bind("<Button>", legend_up)

        else:
            showinfo("Curve colour definition", "Click on the curve you want to extract")
            self.my_canvas.bind("<Button>", colour)

    def extract(self):
        # Data exctraction based on the colour chosen before
        width = self.xmax_pixel - self.xmin_pixel
        height = self.ymax_pixel - self.ymin_pixel
        X = []
        Y = []

        def color_range(r, g, b, p=15):
            R = [i for i in range(r - p, r + p)]
            G = [i for i in range(g - p, g + p)]
            B = [i for i in range(b - p, b + p)]
            return (R, G, B)

        legend_x = [i for i in range(self.x0, self.x1 + 1)]
        legend_y = [i for i in range(self.y0, self.y1 + 1)]

        for x in range(self.xmin_pixel, self.xmax_pixel + 1):
            for y in range(self.ymin_pixel, self.ymax_pixel + 1):
                color = self.img.get(x, y)
                r, g, b = self.R, self.G, self.B
                if (color[0] in color_range(r, g, b)[0]) & (
                    color[1] in color_range(r, g, b)[1]) & (
                    color[2] in color_range(r, g, b)[2]):
                    if not (x in legend_x) & (y in legend_y):
                        X.append(self.Xmin + ((self.Xmax - self.Xmin) / width) * (x -
                                 self.xmin_pixel))
                        Y.append(self.Ymin + ((self.Ymax - self.Ymin) / height) * (
                                (height - y)))

        df = pd.DataFrame(list(zip(X, Y)), columns=['X', 'Y'])
        # If all point to be kept, comment the following line
        if False:
            df = df.drop_duplicates(subset='X', keep='last') # keeps only one point
        else:
            x, y= remove_outliers(df, threshold=0.1, n_points=10, drop=1)
            x_new = np.linspace(np.min(x),np.max(x), 300)
            y_new = np.interp(x_new, x, y)
            df = pd.DataFrame(data = {'X': x_new, 'Y': y_new})
        
        if askyesno('Before we end',"Normalize?"):
            df.Y = df.Y-np.min(df.Y)
            df.Y = df.Y/np.max(df.Y)

        path_save = str(tk.filedialog.askdirectory()) + '/'
        if bool(path_save):
            df.to_csv(path_save + tk.simpledialog.askstring('File name','')
                             + '.csv', sep=';', index=False)
        else:
            self.define()

        self.parent.destroy()


if __name__ == "__main__":
    main = GUI()
    main.mainloop()
