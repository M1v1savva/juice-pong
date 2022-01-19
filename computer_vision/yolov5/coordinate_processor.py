from protocol_clean import default_shot
import numpy as np

class coordinate_processor:

	def __init__(self):
		self.past_x = 0
		self.past_y = 0
		self.past_sign = -1
		self.past_delta = 0


	def drop_past_targets(self):
		self.past_x = 0
		self.past_y = 0
		self.past_sign = -1
		self.past_delta = 0

	def throw_to_coordinates(self, coord=(600, 400)):
		print('calculating the motors configuration...')
		x, y = coord

		print(coord)

		xs = [190, 100, 40, 0, -40]
		ys = [70, 0, -80, -150, -225, -300, -380]

		#xs = [444, 527, 597, 670]
		#ys = [137, 201, 273, 358, 434, 505, 601]

		ds = []
		for i in xs:
			cur = []
			for j in ys:
				cur.append((i, j))
			ds.append(cur)

		for i in ds:
			print(i)

		conf = []

		conf.append([(12, 50),  (7, 50),   (4, 50),  (-2, 50),  (-8, 50),   (-13, 50),  (-16, 50)])
		conf.append([(12, 125), (9, 125),  (4, 125), (-2, 125), (-8, 125),  (-13, 125), (-16, 125)]) #135
		conf.append([(14, 150), (9, 150),  (5, 150), (-2, 150), (-8, 150),  (-13, 150), (-16, 150)])
		conf.append([(16, 160), (10, 170), (6, 170), (-2, 170), (-9, 170),  (-13, 170), (-18, 160)])
		conf.append([(16, 160), (13, 185), (7, 200), (-2, 200), (-11, 200), (-15, 185), (-18, 160)])	

		for i in conf:
			print(i)

		id0 = 0
		for i in range(len(xs)):
			if abs(x - xs[i]) < abs(x - xs[id0]):
				id0 = i

		id1 = 0
		for i in range(len(ys)):
			if ys[i] >= y:
				id1 = i
		print('id1 ' + str(id1))

		y0 = ys[id1]
		y1 = ys[id1 + 1]
		val0 = conf[id0][id1]
		val1 = conf[id0][id1 + 1]
		b, a = val0
		d, c = val1	

		ans0 = 0
		if abs(y - y0) < abs(y - y1):
			ans0 = a
		else:
			ans0 = c
		ans1 = b + (y - y0)/(y1 - y0)*(d - b)

		ans0 = round(ans0)
		ans1 = round(ans1)

		if abs(x - self.past_x) + abs(y - self.past_y) < 15:
			ans0 += self.past_sign * self.past_delta
			self.past_sign = 1 if self.past_sign == -1 else -1
			self.past_delta += 10 if self.past_sign == 1 else 0
		else:
			self.past_x = x
			self.past_y = y
			self.past_sign = 1
			self.past_delta = 10
		

		print(ans0)
		print(ans1)
		
		default_shot(bot_angle=ans1, delay=ans0, wait=5000)

		# id0 = 0
		# id1 = 0

		# for i in range(len(xs)):
		# 	if xs[i] < x:
		# 		id0 = i
		# for i in range(len(ys)):
		# 	if ys[i] < y:
		# 		id1 = i

		# c0 = ds[id0][id1]
		# c1 = ds[id0 + 1][id1 + 1]
		# x0, y0 = c0
		# x1, y1 = c1

		# val0 = conf[id0][id1]
		# val1 = conf[id0 + 1][id1 + 1]
		# b, a = val0
		# d, c = val1

		# print(c0)
		# print(c1)
		# print(val0)
		# print(val1)

		# ans0 = a + (x - x0)/(x1 - x0)*(c - a)
		# ans1 = b + (y - y0)/(y1 - y0)*(d - b)

		# print(ans0)
		# print(ans1)

		# cur = 10000000
		# final_id0 = 0
		# final_id1 = 0
		# for i in range(len(ds)):
		# 	for j in range(len(ds[i])):
		# 		x0, y0 = ds[i][j]
		# 		if abs(x0 - x) + abs(y0 - y) < cur:
		# 			final_id0 = i
		# 			final_id1 = j
		# 			cur = abs(x0 - x) + abs(y0 - y)
		# print(conf[final_id0][final_id1])
		# ans0, ans1 = conf[final_id0][final_id1]
		# print(final_id0)
		# print(final_id1)

		# take closest x - it is the row
		# take 2 closest y
		# now interpolate y and get results
		# then shoot
		# then try to adjust the angle by 1 in each direction

		#default_shot(bot_angle=int(ans0), delay=int(ans1), wait=5000)		
		#default_shot(bot_angle=int(ans1), delay=int(ans0), wait=5000)