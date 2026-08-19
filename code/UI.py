import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np

class DrawingInterface:
    def __init__(self):
        self.grid_size = 28
        self.cell_size = 15

        self.root = tk.Tk()
        self.root.title("Draw Digit")

        self.canvas = tk.Canvas(self.root, 
                                width = self.grid_size * self.cell_size,
                                height= self.grid_size * self.cell_size,
                                bg = 'black')

        self.canvas.pack()

        self.pixel_grid = np.zeros((self.grid_size, self.grid_size))

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.paint)

        btn = tk.Button(self.root, text = "Process Grid", command = self.process_gid)
        btn.pack()
        
        self.root.mainloop()