# Python 3.11
# Rule 30: visualize first 20 states (rows) in a simple Tkinter UI

import tkinter as tk

RULE = 30  # 00011110

def rule30(left: int, center: int, right: int) -> int:
    pattern = (left << 2) | (center << 1) | right  # 0..7
    return (RULE >> pattern) & 1

def generate(rows: int, cols: int) -> list[list[int]]:
    grid = [[0] * cols for _ in range(rows)]
    grid[0][cols // 2] = 0  # single 1 in the middle

    for r in range(1, rows):
        prev = grid[r - 1]
        cur = grid[r]
        for c in range(cols):
            l = prev[c - 1] if c > 0 else 0
            m = prev[c]
            rr = prev[c + 1] if c < cols - 1 else 0
            cur[c] = rule30(l, m, rr)

    return grid

def draw(canvas: tk.Canvas, grid: list[list[int]], cell: int = 16, pad: int = 10) -> None:
    rows = len(grid)
    cols = len(grid[0])
    w = pad * 2 + cols * cell
    h = pad * 2 + rows * cell
    canvas.config(width=w, height=h)
    canvas.delete("all")

    for r, row in enumerate(grid):
        y0 = pad + r * cell
        y1 = y0 + cell
        for c, v in enumerate(row):
            x0 = pad + c * cell
            x1 = x0 + cell
            color = "#111111" if v else "#f2f2f2"
            canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="")

def main():
    rows = 20
    cols = 2 * rows + 1  # fits the triangular growth nicely
    cell = 18

    grid = generate(rows, cols)

    root = tk.Tk()
    root.title("Rule 30 (first 20 states)")
    canvas = tk.Canvas(root, bg="white", highlightthickness=0)
    canvas.pack()

    draw(canvas, grid, cell=cell)
    root.resizable(False, False)
    root.mainloop()

if __name__ == "__main__":
    main()