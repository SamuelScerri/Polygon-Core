from numba import jit, float32
from numba.experimental import jitclass
import numpy as np
import pygame
import typing
from OBJLoader import OBJ, Vertex, Triangle

#config.THREADING_LAYER = 'threadsafe'

WIDTH, HEIGHT = 640, 360

pygame.init()

model = OBJ("Cube.obj")

triangle_mesh = []

for face in model.faces:
	#print(face[0][2])

	v1	= Vertex(model.vertices[face[0][0] - 1][0], model.vertices[face[0][0] - 1][1], model.vertices[face[0][0] - 1][2] - 10)
	v2	= Vertex(model.vertices[face[0][1] - 1][0], model.vertices[face[0][1] - 1][1], model.vertices[face[0][1] - 1][2] - 10)
	v3	= Vertex(model.vertices[face[0][2] - 1][0], model.vertices[face[0][2] - 1][1], model.vertices[face[0][2] - 1][2] - 10)

	triangle_mesh.append(Triangle(v1, v2, v3))
clock = pygame.time.Clock()
font = pygame.font.SysFont("Monospace" , 24 , bold = False)

def get_normalized_coordinates(ts):
	screen = np.array([
		[ts.v1.x, ts.v1.y, ts.v1.z, 1],
		[ts.v2.x, ts.v2.y, ts.v2.z, 1],
		[ts.v3.x, ts.v3.y, ts.v3.z, 1]])

	projection_matrix = np.zeros((4, 4))
	projection_matrix[0][0] = 1 / (np.tan(1.5708 / 2) * (WIDTH / HEIGHT))
	projection_matrix[1][1] = 1 / np.tan(1.5708 / 2)
	projection_matrix[2][2] = (1000 + .1) / (.1 - 1000)
	projection_matrix[2][3] = -1
	projection_matrix[3][2] = (.1 * 1000 * 2) / (.1 - 1000)

	clipped_position = np.array([
		np.matmul(projection_matrix, screen[0]),
		np.matmul(projection_matrix, screen[1]),
		np.matmul(projection_matrix, screen[2])])
	
	normalized = Triangle(
		Vertex(clipped_position[0][0] / clipped_position[0][3], clipped_position[0][1] / clipped_position[0][3], clipped_position[0][2] / clipped_position[0][3]),
		Vertex(clipped_position[1][0] / clipped_position[1][3], clipped_position[1][1] / clipped_position[1][3], clipped_position[1][2] / clipped_position[1][3]),
		Vertex(clipped_position[2][0] / clipped_position[2][3], clipped_position[2][1] / clipped_position[2][3], clipped_position[2][2] / clipped_position[2][3]))

	return normalized

#Here We Compile The Triangle Rasterization Algorithm, This Makes It Super Duper Fast!
#The Reason For Compiling This Is That It's Non-Dependant On Any Library, AKA Pure Computation
@jit(parallel=False, nogil=True, cache=False, nopython=True, fastmath=True)
def process_triangle(ts: Triangle, image_buffer, screen_buffer):
	#Here We Change The World Space Coordinates To Screen Space Coordinates, We Use World Space To Ensure Rendering Will Be Consistent Across All Resolutions

	screen = Triangle(
		Vertex(((ts.v1.x + 1) * WIDTH) / 2, ((-ts.v1.y + 1) * HEIGHT) / 2, 0),
		Vertex(((ts.v2.x + 1) * WIDTH) / 2, ((-ts.v2.y + 1) * HEIGHT) / 2, 0),
		Vertex(((ts.v3.x + 1) * WIDTH) / 2, ((-ts.v3.y + 1) * HEIGHT) / 2, 0))

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

triangle_1 = Triangle(Vertex(-1, -.5, -1), Vertex(0, .5, -1), Vertex(1, -1, -1))
triangle_2 = Triangle(Vertex(-2, -1, 1), Vertex(0, .5, 1), Vertex(0, -1, 1))

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

	triangle_1.translate(Vertex((keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]) / 64, (keys[pygame.K_UP] - keys[pygame.K_DOWN]) / 64, -1 / 64))

	for t in triangle_mesh:
		#vertices = list(model.vertices[face[0][0]])

		t.translate(Vertex((keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]) / 64, (keys[pygame.K_UP] - keys[pygame.K_DOWN]) / 64, 0))
		process_triangle(get_normalized_coordinates(t), image_buffer, screen_buffer)

		#print(vertices)

	
	#process_triangle(get_normalized_coordinates(triangle_1), image_buffer, screen_buffer)
	render_flip(screen_buffer, True)

	screen.blit(font.render("FPS: " + str(clock.get_fps()), False, (255, 255, 255)), (0, 0))
	pygame.display.flip()

	clock.tick()