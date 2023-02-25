from numba.experimental import jitclass

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

		self.v1.z += move_vector.z
		self.v2.z += move_vector.z
		self.v3.z += move_vector.z

class OBJ:
	def __init__(self, filename, swapyz=False):
		"""Loads a Wavefront OBJ file. """
		self.vertices = []
		self.normals = []
		self.texcoords = []
		self.faces = []

		#material = None
		for line in open(filename, "r"):
			if line.startswith('#'): continue
			values = line.split()
			if not values: continue
			if values[0] == 'v':
				v = float(values[1]), float(values[2]), float(values[3])
				self.vertices.append(v)
			elif values[0] == 'vn':
				v = float(values[1]), float(values[2]), float(values[3])
				self.normals.append(v)
			elif values[0] == 'vt':
				v = float(values[1]), float(values[2])
				self.texcoords.append(v)
			elif values[0] == 'f':
				face = []
				texcoords = []
				norms = []
				for v in values[1:]:
					w = v.split('/')
					face.append(int(w[0]))
					if len(w) >= 2 and len(w[1]) > 0:
						texcoords.append(int(w[1]))
					else:
						texcoords.append(0)
					if len(w) >= 3 and len(w[2]) > 0:
						norms.append(int(w[2]))
					else:
						norms.append(0)
				self.faces.append((face, norms, texcoords))