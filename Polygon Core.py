from numba import jit, float32
from numba.experimental import jitclass
import numpy as np
import pygame
import typing
from OBJLoader import OBJ, Vertex, Triangle

WIDTH, HEIGHT = 640, 360
NEAR, FAR = .1, 1000
FOV = 90

pygame.init()

model = OBJ("Cube.obj")
floor = OBJ("Plane.obj")

projection_matrix = np.zeros((4, 4), dtype=np.float32)
projection_matrix[0][0] = 1 / (np.tan((FOV * .01745) / 2) * (WIDTH / HEIGHT))
projection_matrix[1][1] = 1 / np.tan((FOV * .01745) / 2)
projection_matrix[2][2] = (FAR + NEAR) / (NEAR - FAR)
projection_matrix[2][3] = -1
projection_matrix[3][2] = (NEAR * FAR * 2) / (NEAR - FAR)

triangle_mesh = []
floor_mesh = []

for face in model.faces:
	v1	= Vertex(model.vertices[face[0][0] - 1][0], model.vertices[face[0][0] - 1][1], model.vertices[face[0][0] - 1][2] + 20, model.texcoords[face[2][0] - 1][0], 1 - model.texcoords[face[2][0] - 1][1])
	v2	= Vertex(model.vertices[face[0][1] - 1][0], model.vertices[face[0][1] - 1][1], model.vertices[face[0][1] - 1][2] + 20, model.texcoords[face[2][1] - 1][0], 1 - model.texcoords[face[2][1] - 1][1])
	v3	= Vertex(model.vertices[face[0][2] - 1][0], model.vertices[face[0][2] - 1][1], model.vertices[face[0][2] - 1][2] + 20, model.texcoords[face[2][2] - 1][0], 1 - model.texcoords[face[2][2] - 1][1])

	triangle_mesh.append(Triangle(v1, v2, v3))


for face in floor.faces:
	v1	= Vertex(floor.vertices[face[0][0] - 1][0], floor.vertices[face[0][0] - 1][1], floor.vertices[face[0][0] - 1][2] + 20, floor.texcoords[face[2][0] - 1][0], 1 - floor.texcoords[face[2][0] - 1][1])
	v2	= Vertex(floor.vertices[face[0][1] - 1][0], floor.vertices[face[0][1] - 1][1], floor.vertices[face[0][1] - 1][2] + 20, floor.texcoords[face[2][1] - 1][0], 1 - floor.texcoords[face[2][1] - 1][1])
	v3	= Vertex(floor.vertices[face[0][2] - 1][0], floor.vertices[face[0][2] - 1][1], floor.vertices[face[0][2] - 1][2] + 20, floor.texcoords[face[2][2] - 1][0], 1 - floor.texcoords[face[2][2] - 1][1])

	floor_mesh.append(Triangle(v1, v2, v3))

clock = pygame.time.Clock()
font = pygame.font.SysFont("Monospace" , 24 , bold = False)

@jit(parallel=False, nogil=True, cache=False, nopython=True, fastmath=True)
def get_clipped_coordinates(ts, matrix):
	screen = np.array([
		[ts.v1.x, ts.v1.y, ts.v1.z, 1],
		[ts.v2.x, ts.v2.y, ts.v2.z, 1],
		[ts.v3.x, ts.v3.y, ts.v3.z, 1]], dtype=np.float32)

	clipped_1 = np.dot(matrix, screen[0])
	clipped_2 = np.dot(matrix, screen[1])
	clipped_3 = np.dot(matrix, screen[2])

	clipped_triangle = Triangle(
		Vertex(clipped_1[0], clipped_1[1], clipped_1[2], clipped_1[3], 0),
		Vertex(clipped_2[0], clipped_2[1], clipped_2[2], clipped_2[3], 0),
		Vertex(clipped_3[0], clipped_3[1], clipped_3[2], clipped_3[3], 0)
	)

	return clipped_triangle

@jit(parallel=False, nogil=True, cache=False, nopython=True, fastmath=True)
def get_normalized_coordinates(ts, coordinates):
	normalized = Triangle(
		Vertex(coordinates.v1.x / coordinates.v1.tx, coordinates.v1.y / coordinates.v1.tx, coordinates.v1.z / coordinates.v1.tx, ts.v1.tx, ts.v1.ty),
		Vertex(coordinates.v2.x / coordinates.v2.tx, coordinates.v2.y / coordinates.v2.tx, coordinates.v2.z / coordinates.v2.tx, ts.v2.tx, ts.v2.ty),
		Vertex(coordinates.v3.x / coordinates.v3.tx, coordinates.v3.y / coordinates.v3.tx, coordinates.v3.z / coordinates.v3.tx, ts.v3.tx, ts.v3.ty))

	return normalized

#Here We Compile The Triangle Rasterization Algorithm, This Makes It Super Duper Fast!
#The Reason For Compiling This Is That It's Non-Dependant On Any Library, AKA Pure Computation
@jit(parallel=False, nogil=True, cache=False, nopython=True, fastmath=True)
def process_triangle(clipped: Triangle, ts: Triangle, image_buffer, screen_buffer, depth_buffer, matrix):
	#Here We Change The World Space Coordinates To Screen Space Coordinates, We Use World Space To Ensure Rendering Will Be Consistent Across All Resolutions

	screen = Triangle(
		Vertex(((ts.v1.x + 1) * WIDTH) / 2, ((-ts.v1.y + 1) * HEIGHT) / 2, ts.v1.z, ts.v1.tx, ts.v1.ty),
		Vertex(((ts.v2.x + 1) * WIDTH) / 2, ((-ts.v2.y + 1) * HEIGHT) / 2, ts.v2.z, ts.v2.tx, ts.v2.ty),
		Vertex(((ts.v3.x + 1) * WIDTH) / 2, ((-ts.v3.y + 1) * HEIGHT) / 2, ts.v3.z, ts.v3.tx, ts.v3.ty))

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

				depth = w * screen.v1.z + s * screen.v2.z + t * screen.v3.z
				print(depth)

				#Here We Convert The RGB Value To A Color Integer, Then It Is Assigned To The Screen Buffer
				if depth > depth_buffer[x][y]:
					#Thanks To The Barycentric Coordinates, We Could Interpolate Colour Values Between Every Point
					r = int(w * 255 + s * 0 + t * 0)
					g = int(w * 0 + s * 255 + t * 0)
					b = int(w * 0 + s * 0 + t * 255)

					if image_buffer == None:
						screen_buffer[x][y] = (r << 16) + (g << 8) + b
						depth_buffer[x][y] = depth

					else:
						#We Get The Color Coordinates From The Image Buffer
						uvx = int((w * screen.v1.tx + s * screen.v2.tx + t * screen.v3.tx) * image_buffer.shape[0])
						uvy = int((w * screen.v1.ty + s * screen.v2.ty + t * screen.v3.ty) * image_buffer.shape[1])
					
						#Here We Convert To RGB To Implement Additive Mixing
						uvr = ((image_buffer[uvx][uvy] >> 16) & 0xff)
						uvg = ((image_buffer[uvx][uvy] >> 8) & 0xff)
						uvb = ((image_buffer[uvx][uvy]) & 0xff)

						#Responsible For Mixing The Two Colors
						final_r = int((uvr * r) / 255)
						final_g = int((uvg * g) / 255)
						final_b = int((uvb * b) / 255)

						screen_buffer[x][y] = (final_r << 16) + (final_g << 8) + final_b
						depth_buffer[x][y] = depth

@jit(parallel=False, nogil=True, cache=False, nopython=True, fastmath=True)
def clamp(num, min_value, max_value):
	return max(min(num, max_value), min_value)

triangle_1 = Triangle(Vertex(-1, -.5, 5, 0, .5), Vertex(0, .5, 5, 0, 0), Vertex(1, -1, 5, .5, .5))
triangle_2 = Triangle(Vertex(-2, -1, 1), Vertex(0, .5, 1), Vertex(0, -1, 1))

#Here We Render The Final Image On To The Screen
def render_flip(screen_buffer, clear = True):
	pygame.surfarray.blit_array(pygame.display.get_surface(), screen_buffer)

	#This Will Clear Everything & Avoid The Ghost Effect, This Should Not Be Used In Closed Rooms As It Is A Waste Of Processing
	if clear:
		screen_buffer.fill(0)
		depth_buffer.fill(0)

@jit(parallel=False, nogil=True, cache=False, nopython=True, fastmath=True)
def render_model(model, image_buffer, screen_buffer, depth_buffer, matrix):
	for i in range(len(model)):
		clipped_coordinates = get_clipped_coordinates(model[i], matrix)

		clipped_coordinates.v1.tx = clamp(clipped_coordinates.v1.tx, -9999, -0.1)
		clipped_coordinates.v2.tx = clamp(clipped_coordinates.v2.tx, -9999, -0.1)
		clipped_coordinates.v3.tx = clamp(clipped_coordinates.v3.tx, -9999, -0.1)

		if clipped_coordinates.v1.tx < 0 and clipped_coordinates.v2.tx < 0 and clipped_coordinates.v3.tx < 0:
			normalized_coordinates = get_normalized_coordinates(model[i], clipped_coordinates)
			process_triangle(clipped_coordinates, normalized_coordinates, image_buffer, screen_buffer, depth_buffer, matrix)


@jit(parallel=False, nogil=True, cache=False, nopython=True, fastmath=True)
def translate_model(model, x, y, z):
	for i in range(len(model)):
		model[i].translate(x, y, z)

running = True

screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.SCALED, vsync=True)
pygame.display.set_caption("Polygon Core")

image = pygame.image.load("Brick.bmp").convert()
crate_image = pygame.image.load("Texture.png").convert()

#The Renderer Only Works With Image Pixel Data, Here We Reference As To Avoid Large Memory Allocations
image_buffer = pygame.surfarray.pixels2d(image)
crate_image_buffer = pygame.surfarray.pixels2d(crate_image)

#Here We Create The Screen Buffer, This Is Useful As To Avoid Calling The Blit Function Multiple Times
screen_buffer = np.zeros((WIDTH, HEIGHT), dtype=np.int32)
depth_buffer = np.zeros((WIDTH, HEIGHT), dtype=np.float32)

while running:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			running = False

	keys = pygame.key.get_pressed()

	translate_model(triangle_mesh, (keys[pygame.K_LEFT] - keys[pygame.K_RIGHT]) / 8, (keys[pygame.K_DOWN] - keys[pygame.K_UP]) / 8, (keys[pygame.K_w] - keys[pygame.K_s]) / 8)
	render_model(triangle_mesh, image_buffer, screen_buffer, depth_buffer, projection_matrix)
	
	render_flip(screen_buffer, True)

	screen.blit(font.render("FPS: " + str(clock.get_fps()), False, (255, 255, 255)), (0, 0))
	pygame.display.flip()

	clock.tick()