from numba import jit, float32
from numba.experimental import jitclass
import numpy as np
import pygame
import typing

#config.THREADING_LAYER = 'threadsafe'

WIDTH, HEIGHT = 640, 360

pygame.init()

clock = pygame.time.Clock()
font = pygame.font.SysFont("Monospace" , 24 , bold = False)

@jitclass
class Vertex:
	x: float
	y: float
	z: float

	def __init__(self, x: float, y: float, z: float = 0):
		self.x = x
		self.y = y
		self.z = z

@jitclass
class Triangle:
	v1: Vertex
	v2: Vertex
	v3: Vertex

	def __init__(self, v1: Vertex, v2: Vertex, v3: Vertex):
		self.v1 = v1
		self.v2 = v2
		self.v3 = v3

	def translate(self, move_vector: Vertex):
		self.v1.x += move_vector.x
		self.v2.x += move_vector.x
		self.v3.x += move_vector.x

		self.v1.y += move_vector.y
		self.v2.y += move_vector.y
		self.v3.y += move_vector.y

#Here We Compile The Triangle Rasterization Algorithm, This Makes It Super Duper Fast!
#The Reason For Compiling This Is That It's Non-Dependant On Any Library, AKA Pure Computation
@jit(parallel=False, nogil=True, cache=False, nopython=True, fastmath=True)
def process_triangle(ts: Triangle, image_buffer, screen_buffer):
	#Here We Change The World Space Coordinates To Screen Space Coordinates, We Use World Space To Ensure Rendering Will Be Consistent Across All Resolutions
	screen = Triangle(
		Vertex(((ts.v1.x * screen_buffer.shape[1] / screen_buffer.shape[0] + 1) * screen_buffer.shape[0]) / 2, ((-ts.v1.y + 1) * screen_buffer.shape[1]) / 2, 0),
		Vertex(((ts.v2.x * screen_buffer.shape[1] / screen_buffer.shape[0] + 1) * screen_buffer.shape[0]) / 2, ((-ts.v2.y + 1) * screen_buffer.shape[1]) / 2, 0),
		Vertex(((ts.v3.x * screen_buffer.shape[1] / screen_buffer.shape[0] + 1) * screen_buffer.shape[0]) / 2, ((-ts.v3.y + 1) * screen_buffer.shape[1]) / 2, 0))

	vs1 = (screen.v2.x - screen.v1.x, screen.v2.y - screen.v1.y)
	vs2 = (screen.v3.x - screen.v1.x, screen.v3.y - screen.v1.y)
	span_product = vs1[0] * vs2[1] - vs1[1] * vs2[0]

	#We Clamp This To Ensure Anything Outside The Window Boundaries Won't Get Rendered!
	max_x = clamp(int(max(screen.v1.x, max(screen.v2.x, screen.v3.x))), 0, WIDTH)
	min_x = clamp(int(min(screen.v1.x, min(screen.v2.x, screen.v3.x))), 0, WIDTH)
	max_y = clamp(int(max(screen.v1.y, max(screen.v2.y, screen.v3.y))), 0, HEIGHT)
	min_y = clamp(int(min(screen.v1.y, min(screen.v2.y, screen.v3.y))), 0, HEIGHT)

	#We Don't Loop Through The Entire Buffer As There Is No Need
	for x in range(min_x, max_x):
		for y in range(min_y, max_y):
			q = (x - screen.v1.x, y - screen.v1.y)

			s = (q[0] * vs2[1] - q[1] * vs2[0]) / span_product
			t = (vs1[0] * q[1] - vs1[1] * q[0]) / span_product

			#Essentially This Means That The Coordinates Are In The Triangle Coordinates
			if s > 0 and t > 0 and s + t <= 1:

				#We Get The Third Barycentric Coordinate Here
				w = (1 - s - t)

				#Thanks To The Barycentric Coordinates, We Could Interpolate Colour Values Between Every Point
				r = int(w * 255 + s * 0 + t * 0)
				g = int(w * 0 + s * 255 + t * 0)
				b = int(w * 0 + s * 0 + t * 255)

				if image_buffer is None:
					screen_buffer[x][y] = (r << 16) + (g << 8) + b
				else:
					#We Get The Color Coordinates From The Image Buffer
					uvx = int((w * 0 + s * .5 + t * .5) * image_buffer.shape[0])
					uvy = int((w * 0 + s * 0 + t * .5) * image_buffer.shape[1])

					#Here We Convert To RGB To Implement Additive Mixing
					uvr = ((image_buffer[uvx][uvy] >> 16) & 0xff)
					uvg = ((image_buffer[uvx][uvy] >> 8) & 0xff)
					uvb = ((image_buffer[uvx][uvy]) & 0xff)

					#Responsible For Mixing The Two Colors
					final_r = int((uvr * r) / 255)
					final_g = int((uvg * g) / 255)
					final_b = int((uvb * b) / 255)

					#Here We Convert The RGB Value To A Color Integer, Then It Is Assigned To The Screen Buffer
					screen_buffer[x][y] = (final_r << 16) + (final_g << 8) + final_b

@jit(parallel=False, nogil=True, cache=False, nopython=True, fastmath=True)
def clamp(num, min_value, max_value):
	return max(min(num, max_value), min_value)

triangle_1 = Triangle(Vertex(-1, -.5, 0), Vertex(0, .5, 0), Vertex(1, -1, 0))
triangle_2 = Triangle(Vertex(-2, -1, 0), Vertex(0, .5, 0), Vertex(0, -1, 0))

#Here We Render The Final Image On To The Screen
def render_flip(screen_buffer, clear = True):
	pygame.surfarray.blit_array(pygame.display.get_surface(), screen_buffer)

	#This Will Clear Everything & Avoid The Ghost Effect, This Should Not Be Used In Closed Rooms As It Is A Waste Of Processing
	if clear:
		screen_buffer.fill(0)

running = True

screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.SCALED, vsync=False)
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

	triangle_1.translate(Vertex((keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]) / 256, (keys[pygame.K_UP] - keys[pygame.K_DOWN]) / 256))

	process_triangle(triangle_1, image_buffer, screen_buffer)
	process_triangle(triangle_2, image_buffer, screen_buffer)
	render_flip(screen_buffer, True)

	screen.blit(font.render("FPS: " + str(clock.get_fps()), False, (255, 255, 255)), (0, 0))
	pygame.display.flip()

	clock.tick()