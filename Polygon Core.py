from numba import njit, cuda, jit, vectorize, prange, config, threading_layer, float32
from numba.experimental import jitclass
import numpy as np
import sys
import time
import pygame

#config.THREADING_LAYER = 'threadsafe'

WIDTH, HEIGHT = 640, 360

pygame.init()

clock = pygame.time.Clock()
font = pygame.font.SysFont("Monospace" , 24 , bold = False)

#Here We Compile The Triangle Rasterization Algorithm, This Makes It Super Duper Fast!
#The Reason For Compiling This Is That It's Non-Dependant On Any Library, AKA Pure Computation
@jit(parallel=False, nogil=True, cache=True, nopython=True, fastmath=True)
def process_triangle(ts, image_buffer, screen_buffer):
	screen = (
		(((ts[0][0] * screen_buffer.shape[1] / screen_buffer.shape[0] + 1) * screen_buffer.shape[0]) / 2, ((-ts[0][1] + 1) * screen_buffer.shape[1]) / 2),
		(((ts[1][0] * screen_buffer.shape[1] / screen_buffer.shape[0] + 1) * screen_buffer.shape[0]) / 2, ((-ts[1][1] + 1) * screen_buffer.shape[1]) / 2),
		(((ts[2][0] * screen_buffer.shape[1] / screen_buffer.shape[0] + 1) * screen_buffer.shape[0]) / 2, ((-ts[2][1] + 1) * screen_buffer.shape[1]) / 2))

	vs1 = (screen[1][0] - screen[0][0], screen[1][1] - screen[0][1])
	vs2 = (screen[2][0] - screen[0][0], screen[2][1] - screen[0][1])
	span_product = vs1[0] * vs2[1] - vs1[1] * vs2[0]

	#We Clamp This To Ensure Anything Outside The Window Boundaries Won't Get Rendered!
	max_x = clamp(int(max(screen[0][0], max(screen[1][0], screen[2][0]))), 0, WIDTH)
	min_x = clamp(int(min(screen[0][0], min(screen[1][0], screen[2][0]))), 0, WIDTH)
	max_y = clamp(int(max(screen[0][1], max(screen[1][1], screen[2][1]))), 0, HEIGHT)
	min_y = clamp(int(min(screen[0][1], min(screen[1][1], screen[2][1]))), 0, HEIGHT)

	#We Don't Loop Through The Entire Buffer As There Is No Need
	for x in range(min_x, max_x):
		for y in range(min_y, max_y):
			q = (x - screen[0][0], y - screen[0][1])

			s = (q[0] * vs2[1] - q[1] * vs2[0]) / span_product
			t = (vs1[0] * q[1] - vs1[1] * q[0]) / span_product

			#Essentially This Means That The Coordinates Are In The Triangle Coordinates
			if s > 0 and t > 0 and s + t <= 1:
				w = (1 - s - t)

				r = int(w * 255 + s * 0 + t * 0)
				g = int(w * 0 + s * 255 + t * 0)
				b = int(w * 0 + s * 0 + t * 255)

				if image_buffer is None:
					screen_buffer[x][y] = (r << 16) + (g << 8) + b
				else:
					#We Get The Color Coordinates From The Image Buffer
					uvx = int((w * 0 + s * 1 + t * 1) * image_buffer.shape[0])
					uvy = int((w * 0 + s * 0 + t * 1) * image_buffer.shape[1])

					#Here We Convert To RGB To Implement Additive Mixing
					uvr = ((image_buffer[uvx][uvy] >> 16) & 0xff)
					uvg = ((image_buffer[uvx][uvy] >> 8) & 0xff)
					uvb = ((image_buffer[uvx][uvy]) & 0xff)

					#Responsible For Mixing The Two Colors
					final_r = int((uvr * r) / 255)
					final_g = int((uvg * g) / 255)
					final_b = int((uvb * b) / 255)

					screen_buffer[x][y] = (final_r << 16) + (final_g << 8) + final_b

@jit(parallel=False, nogil=True, cache=True, nopython=True, fastmath=True)
def clamp(num, min_value, max_value):
   return max(min(num, max_value), min_value)

def render_flip(screen_buffer, clear = False):
	pygame.surfarray.blit_array(pygame.display.get_surface(), screen_buffer)

	if clear:
		screen_buffer.fill(0)

running = True

triangle_1 = ((-1, -.5), (0, .5), (1, -.5))

screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.SCALED | pygame.FULLSCREEN, vsync=True)
pygame.display.set_caption("Polygon Core")

image = pygame.image.load("Brick.bmp").convert()

#The Renderer Only Works With Image Pixel Data, Here We Reference As To Avoid Large Memory Allocations
image_buffer = pygame.surfarray.pixels2d(image)

#Here We Create The Screen Buffer, This Is Useful As To Avoid Calling The Blit Function Multiple Times
screen_buffer = np.zeros((WIDTH, HEIGHT), dtype=np.int32)

while running:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			running = False

	keys = pygame.key.get_pressed()

	triangle_1 = ((triangle_1[0][0] + (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]) / 64, triangle_1[0][1] + (keys[pygame.K_UP] - keys[pygame.K_DOWN]) / 64),
	(triangle_1[1][0] + (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]) / 64, triangle_1[1][1] + (keys[pygame.K_UP] - keys[pygame.K_DOWN]) / 64),
	(triangle_1[2][0] + (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]) / 64, triangle_1[2][1] + (keys[pygame.K_UP] - keys[pygame.K_DOWN]) / 64))

	start = time.time()

	process_triangle(triangle_1, image_buffer, screen_buffer)
	render_flip(screen_buffer, True)

	screen.blit(font.render("FPS: " + str(clock.get_fps()), False, (255, 255, 255)), (0, 0))
	pygame.display.flip()

	clock.tick()