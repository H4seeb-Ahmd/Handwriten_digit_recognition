import tkinter as tk
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

        process_btn = tk.Button(self.root, text = "Process Grid", command = self.process_grid)
        process_btn.pack()
        clear_btn = tk.Button(self.root, text = "Clear", command = self.clear)
        clear_btn.pack()

        self.root.mainloop()

    def paint(self, event):

        col = event.x // self.cell_size
        row = event.y // self.cell_size

        if 0 <= col < self.grid_size and 0 <= row < self.grid_size:
            self.pixel_grid[row, col] = 255

            x1 = col * self.cell_size
            y1 = row * self.cell_size
            x2 = x1 + self.cell_size
            y2 = y1 + self.cell_size
            self.canvas.create_rectangle(x1, y1, x2, y2, fill='white', outline='white')

    def clear(self):
        self.canvas.delete("all")
        self.pixel_grid.fill(0)

    def process_grid(self):
        print("Grid captured! Current shape is:", self.pixel_grid.shape)

DrawingInterface()