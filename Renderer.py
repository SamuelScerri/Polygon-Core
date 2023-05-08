import pygame
import numba
import numpy

SIZE = (1280, 720)

pygame.init()

screen = pygame.display.set_mode(SIZE, pygame.SCALED, vsync=True)
pygame.display.set_caption("Renderer")

screen_buffer = numpy.zeros(SIZE, dtype=numpy.int32)

NEAR = .5
FAR = 100.0
FOV = 45

aspect_ratio = SIZE[0] / SIZE[1]
top = numpy.tan((FOV * .01745) / 2) * NEAR
bottom = -top
right = top * aspect_ratio
bottom = -top * aspect_ratio
left = bottom

#The Projection Matrix Is Based Off The OpenGL Perspective Projection Matrix
projection_matrix = numpy.zeros((4, 4), dtype=numpy.float32)
projection_matrix[0][0] = (2 * NEAR) / (right - left)
projection_matrix[1][1] = (2 * NEAR) / (top - bottom)

projection_matrix[2][0] = (right + left) / (right - left)
projection_matrix[2][1] = (top + bottom) / (top - bottom)
projection_matrix[2][2] = -1 * ((FAR + NEAR) / (FAR - NEAR))
projection_matrix[2][3] = -1
projection_matrix[3][2] = -1 * ((2 * FAR * NEAR) / (FAR - NEAR))

running = True

#triangle_test = (
#	(0.0, 0.5, -4.0, 1.0),
#	(-0.5, -0.5, -4.0, 1.0),
#	(0.5, -0.5, -4.0, 1.0)
#)

triangle_test = (
	(0.1, 0.1, -5.0, 1.0),
	(0.1, -0.1, -5.5, 1.0),
	(0.1, -0.1, -4.5, 1.0),
)

uv = (
	(-.2, -.2),
	(.2, -.2),
	(.2, .2)
)

@numba.jit(parallel=False, nogil=True, cache=True, nopython=True, fastmath=True)
def lerp(a, b, t):
	#print(a, b)

	return a * (1.0 - t) + b * t

@numba.jit(parallel=False, nogil=True, cache=True, nopython=True, fastmath=True)
def get_clipped_coordinates(vertex_coordinate, matrix):
	matrix_coordinate = numpy.array(vertex_coordinate, dtype=numpy.float32)

	clipped_1 = numpy.dot(matrix, matrix_coordinate[0])
	clipped_2 = numpy.dot(matrix, matrix_coordinate[1])
	clipped_3 = numpy.dot(matrix, matrix_coordinate[2])

	return [
		[clipped_1[0], clipped_1[1], clipped_1[2], clipped_1[3]],
		[clipped_2[0], clipped_2[1], clipped_2[2], clipped_2[3]],
		[clipped_3[0], clipped_3[1], clipped_3[2], clipped_3[3]]
	]

@numba.jit(parallel=False, nogil=True, cache=True, nopython=True, fastmath=True)
def get_normalized_coordinate(vertex_coordinate):


	normalized_1 = (vertex_coordinate[0] / vertex_coordinate[3], vertex_coordinate[1] / vertex_coordinate[3], vertex_coordinate[2] / vertex_coordinate[3])

	return normalized_1

@numba.jit(parallel=False, nogil=True, cache=True, nopython=True, fastmath=True)
def lerp_vertex(vertex, target, factor):
	return [
		lerp(vertex[0], target[0], factor),
		lerp(vertex[1], target[1], factor),
		lerp(vertex[2], target[2], factor),
		lerp(vertex[3], target[3], factor)
	]

@numba.jit(parallel=False, nogil=True, cache=True, nopython=False, fastmath=True)
def clip_triangle(clipped_coordinate, uv_coordinate, axis):
	vertex_list = []
	uv_list = []

	previous_vertex = clipped_coordinate[len(clipped_coordinate) - 1]
	previous_inside = previous_vertex[axis] <= previous_vertex[3]
	previous_uv = uv_coordinate[len(clipped_coordinate) - 1]

	for index in range(len(clipped_coordinate)):
		current_vertex = clipped_coordinate[index]
		current_inside = current_vertex[axis] <= current_vertex[3]

		if current_inside ^ previous_inside:
			factor = (previous_vertex[3] - previous_vertex[axis]) / (
				(previous_vertex[3] - previous_vertex[axis]) -
				(current_vertex[3] - current_vertex[axis])
			)

			vertex_list.append([
				lerp(previous_vertex[0], current_vertex[0], factor),
				lerp(previous_vertex[1], current_vertex[1], factor),
				lerp(previous_vertex[2], current_vertex[2], factor),
				lerp(previous_vertex[3], current_vertex[3], factor)
			])

			uv_list.append((
				lerp(previous_uv[0], uv_coordinate[index][0], factor),
				lerp(previous_uv[1], uv_coordinate[index][1], factor)
			))

		
		if current_inside:
			vertex_list.append(current_vertex)
			uv_list.append(uv_coordinate[index])

		previous_vertex = current_vertex
		previous_inside = current_inside
		previous_uv = uv_coordinate[index]	

	return vertex_list, uv_list

@numba.jit(parallel=False, nogil=True, cache=True, nopython=False, fastmath=False)
def process_polygon(vertex_coordinate, uv_coordinate, texture, screen_buffer, index):
	computed_triangle = (vertex_coordinate[0], vertex_coordinate[index + 1], vertex_coordinate[index + 2])
	computed_uv = (uv_coordinate[0], uv_coordinate[index + 1], uv_coordinate[index + 2])

	process_triangle(computed_triangle, computed_uv, texture, screen_buffer)

	if index + 3 < len(vertex_coordinate):
		process_polygon(vertex_coordinate, uv_coordinate, texture, screen_buffer, index + 1)

@numba.jit(parallel=False, nogil=True, cache=True, nopython=False, fastmath=False)
def render_triangle(vertex_coordinate, uv_coordinate, texture, screen_buffer):
	vertex_list = []
	uv_list = []

	normalized_coordinates = []

	clipped_coordinate = get_clipped_coordinates(vertex_coordinate, projection_matrix)
	#Thanks To The Clipped Coordinate We Can See Which Exact Coordinate The Triangle Will Leave The Screen
	#With This We Will Now Clip The Triangle Into Multiple Triangles If Its Intersecting With The Clipping Coordinate
	
	vertex_list, uv_list = clip_triangle(clipped_coordinate, uv_coordinate, 0)

	if len(vertex_list) > 0:
		vertex_list, uv_list = clip_triangle(vertex_list, uv_list, 1)

	if len(vertex_list) > 0:
		vertex_list, uv_list = clip_triangle(vertex_list, uv_list, 2)

	if len(vertex_list) > 0:
		for coordinate in vertex_list:
			normalized_coordinates.append(get_normalized_coordinate(coordinate))

		#Simple Polygon Rendering
		for index in range(len(normalized_coordinates) - 2):
			second_triangle = (
				normalized_coordinates[0],
				normalized_coordinates[index + 1],
				normalized_coordinates[index + 2]
			)

			second_uv = (
				uv_list[0],
				uv_list[index + 1],
				uv_list[index + 2]
			)

			process_triangle(second_triangle, second_uv, texture, screen_buffer)

@numba.jit(parallel=False, nogil=True, cache=True, nopython=False, fastmath=True)
def process_triangle(vertex_coordinate, uv_coordinate, texture, screen_buffer):
	screen_space_coordinate = (
		(((vertex_coordinate[0][0] + 1) * screen_buffer.shape[0]) / 2, ((-vertex_coordinate[0][1] + 1) * screen_buffer.shape[1]) / 2),
		(((vertex_coordinate[1][0] + 1) * screen_buffer.shape[0]) / 2, ((-vertex_coordinate[1][1] + 1) * screen_buffer.shape[1]) / 2),
		(((vertex_coordinate[2][0] + 1) * screen_buffer.shape[0]) / 2, ((-vertex_coordinate[2][1] + 1) * screen_buffer.shape[1]) / 2)
	)

	vertex_span_1 = (screen_space_coordinate[1][0] - screen_space_coordinate[0][0], screen_space_coordinate[1][1] - screen_space_coordinate[0][1])
	vertex_span_2 = (screen_space_coordinate[2][0] - screen_space_coordinate[0][0], screen_space_coordinate[2][1] - screen_space_coordinate[0][1])

	span_product = vertex_span_1[0] * vertex_span_2[1] - vertex_span_1[1] * vertex_span_2[0]

	for x in range(0, screen_buffer.shape[0]):
		for y in range(0, screen_buffer.shape[1]):
			q = (x - screen_space_coordinate[0][0], y - screen_space_coordinate[0][1])

			s = (q[0] * vertex_span_2[1] - q[1] * vertex_span_2[0]) / span_product
			t = (vertex_span_1[0] * q[1] - vertex_span_1[1] * q[0]) / span_product

			w = 1 - s - t

			if s > 0 and t > 0 and s + t <= 1:
				uvx = int((w * uv_coordinate[0][0] + s * uv_coordinate[1][0] + t * uv_coordinate[2][0]) * texture.shape[0])
				uvy = int((w * uv_coordinate[0][0] + s * uv_coordinate[1][1] + t * uv_coordinate[2][1]) * texture.shape[1])	

				screen_buffer[x][y] = texture[uvx][uvy]	

image = pygame.image.load("Brick.bmp").convert()
image_buffer = pygame.surfarray.pixels2d(image)

while running:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			running = False

	keys = pygame.key.get_pressed()

	triangle_test = (
		(
			triangle_test[0][0] + (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]) * .01,
			triangle_test[0][1] + (keys[pygame.K_UP] - keys[pygame.K_DOWN]) * .01,
			triangle_test[0][2] + (keys[pygame.K_s] - keys[pygame.K_w]) * .01,
			triangle_test[0][3]
		),

		(
			triangle_test[1][0] + (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]) * .01,
			triangle_test[1][1] + (keys[pygame.K_UP] - keys[pygame.K_DOWN]) * .01,
			triangle_test[1][2] + (keys[pygame.K_s] - keys[pygame.K_w]) * .01,
			triangle_test[1][3]
		),

		(
			triangle_test[2][0] + (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]) * .01,
			triangle_test[2][1] + (keys[pygame.K_UP] - keys[pygame.K_DOWN]) * .01,
			triangle_test[2][2] + (keys[pygame.K_s] - keys[pygame.K_w]) * .01,
			triangle_test[2][3]
		),
	)

	render_triangle(triangle_test, uv, image_buffer, screen_buffer)

	pygame.surfarray.blit_array(pygame.display.get_surface(), screen_buffer)
	pygame.display.flip()

	NEAR += .01
	
	screen_buffer.fill(0)

pygame.quit()