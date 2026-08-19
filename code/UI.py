import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw
import DIGIT_RECOGNIZING_MODELS as Models

class DrawingInterface:
    def __init__(self, model):
        self.model = model()
        self.grid_size = 28
        self.cell_size = 10
        self.radius = 12

        self.pixel_grid = np.zeros((self.grid_size, self.grid_size))

        self.width = self.grid_size * self.cell_size
        self.height = self.grid_size * self.cell_size

        self.root = tk.Tk()
        self.root.title("Digit_Recognition")

        self.prediction = tk.Label(self.root, text = "(prediction)")

        self.prediction.pack()

        self.canvas = tk.Canvas(self.root, 
                                width = self.width,
                                height = self.height,
                                bg = 'black')

        self.canvas.pack()


        self.image = Image.new("L", (self.width, self.height), color='black')
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)

        

        process_btn = tk.Button(self.root, text = "Process Grid", command = self.process_grid)
        process_btn.pack()
        clear_btn = tk.Button(self.root, text = "Clear", command = self.clear)
        clear_btn.pack()

        self.root.mainloop()

    def paint(self, event):

        x1, y1 = event.x - self.radius, event.y - self.radius
        x2, y2 = event.x + self.radius, event.y + self.radius
        self.canvas.create_oval(x1, y1, x2, y2, fill='white', outline='white')
        self.draw.ellipse([x1, y1, x2, y2], fill='white')

    def clear(self):
        self.canvas.delete("all")
        self.prediction.config(text = f"(prediction)")
        self.pixel_grid.fill(0)
        self.image = Image.new("L", (self.width, self.height), color='black')
        self.draw = ImageDraw.Draw(self.image)

    def process_grid(self):
        self.small_img = self.image.resize((self.grid_size, self.grid_size), Image.Resampling.LANCZOS)
        self.pixel_grid = np.array(self.small_img)

        self.flat_image = self.pixel_grid.reshape(1, 784)
        
        result = self.model.model_prediction(self.flat_image)
        self.prediction.config(text = f"Predicted Digit: {result[0]}")
        
DrawingInterface(Models.KNN)