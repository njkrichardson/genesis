# http://numba.pydata.org/numba-doc/latest/user/examples.html

import matplotlib.pyplot as plt
import numpy as np
from numba import jit


@jit
def mandel(x, y, max_iters):
    i = 0
    c = complex(x, y)
    z = 0.0j

    for i in range(max_iters):
        z = z * z + c

        if z.real * z.real + z.imag * z.imag >= 4:
            return i

    return 255


@jit
def create_fractal(min_x, max_x, min_y, max_y, image, iters):
    height, width = image.shape

    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height

    for x in range(width):
        real = min_x + x * pixel_size_x

        for y in range(height):
            imag = min_y + y * pixel_size_y

            color = mandel(real, imag, iters)
            image[y, x] = color

    return image


if __name__ == "__main__":
    image = np.zeros((500 * 2, 750 * 2), dtype=np.uint8)
    create_fractal(-2.0, 1.0, -1.0, 1.0, image, 20)

    plt.jet()
    plt.imshow(image)
    plt.show()
