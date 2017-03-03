import cv2
import numpy as np
import random
from keras import backend as K

import matplotlib.pylab as plt

def load_class_mapping(basedir='data', type='theme'):
	"""load the mapping of ids to theme categories"""
	d = {}
	f = open("{}/key-{}.txt".format(basedir, type) , 'r')
	for line in f:
		clz, color = line.split("\t")
		d[int(color)] = clz

	return d

def load_image_pair(gridref, basedir='data', clztype='theme', vistype='3band'):
	"""load the sensor and classified images corresponding to a grid ref (XXEENN)."""
	img = cv2.imread("{0}/{1}/{2}.TIF".format(basedir, vistype, gridref), cv2.IMREAD_ANYCOLOR)
	clz = cv2.imread("{0}/{1}/{2}-{1}.png".format(basedir, clztype, gridref), cv2.IMREAD_ANYDEPTH)
	return img, clz

def coord_generator(width, height, patchsize, patchstep):
	"""Generate scan-ordered coordinates for sampling patches from an image"""
	for y in xrange(0, height - patchsize, patchstep):
		for x in xrange(0, width - patchsize, patchstep):
			yield x, y

def shuffled_coord_generator(width, height, patchsize, patchstep):
	"""Generate shuffled coordinates for sampling patches from an image"""
	# this is slightly memory hungry but allows for proper shuffling
	sample_locations = []
	for y in xrange(0, height - patchsize, patchstep):
		for x in xrange(0, width - patchsize, patchstep):
			sample_locations.append((x,y))

	random.shuffle(sample_locations)

	for x, y in sample_locations:
		yield x, y

def generate_labelled_patches(gridrefs, patchsize, patchstep=1, batch_size=32, shuffle=False, basedir='data', clztype='theme', vistype='3band'):
	"""Generate batches of labelled patches"""
	mapping = load_class_mapping(basedir=basedir, type=clztype)
	mapping_keys = [ k for k in mapping ]

	dim_ordering = K.image_dim_ordering()
	
	batch_clz = None
	batch_img = None
	batch_idx = 0
	while True:
		if shuffle:
			random.shuffle(gridrefs)

		for gridref in gridrefs:
			img, clz = load_image_pair(gridref, basedir=basedir, clztype=clztype, vistype=vistype)

			img = img.astype(K.floatx())
			img = img / 255

			width, height, depth = img.shape

			if batch_clz == None:
				batch_clz = np.zeros((batch_size, len(mapping)))
				batch_img = np.zeros((batch_size, patchsize, patchsize, depth))
				
			if shuffle:
				sample_locations = shuffled_coord_generator
			else:
				sample_locations = coord_generator

			for x,y in sample_locations(width, height, patchsize, patchstep):
				patch = img[y:y+patchsize, x:x+patchsize, :]
				theclz = clz[y+patchsize/2, x+patchsize/2]

				batch_img[batch_idx, :, :, :] = patch
				batch_clz[batch_idx, :] = 0
				batch_clz[batch_idx][mapping_keys.index(theclz)] = 1 
				batch_idx = batch_idx + 1

				if batch_idx == batch_size:
					batch_idx = 0

					if dim_ordering == 'th':
						yield batch_img.transpose(0, 3, 1, 2), batch_clz
					else:
						yield batch_img, batch_clz

def extract_roi(img, clz, subcoords):
	img = img[subcoords[0][0]:subcoords[1][0], subcoords[0][1]:subcoords[1][1], :]
	clz = clz[subcoords[0][0]:subcoords[1][0], subcoords[0][1]:subcoords[1][1]]
	return img, clz

def load_labelled_patches(gridrefs, patchsize, patchstep=1, subcoords=None, limit=None, shuffle=False, basedir='data', clztype='theme', vistype='3band'):
	"""Load a set of labelled patches into memory"""
	
	if subcoords != None:
		assert len(gridrefs) == 1, 'subcoords can only be used with a single image'
		assert limit == None, 'subcoords and limit are mutually exclusive'
		assert shuffle == False, 'Shuffling when using subcoords isn\'t supported'

	mapping = load_class_mapping(basedir=basedir, type=clztype)
	mapping_keys = [ k for k in mapping ]

	dim_ordering = K.image_dim_ordering()
	
	batch_clz = None
	batch_img = None
	batch_idx = 0

	if shuffle:
		random.shuffle(gridrefs)

	for gridref in gridrefs:
		img, clz = load_image_pair(gridref, basedir=basedir, clztype=clztype, vistype=vistype)

		img = img.astype(K.floatx())
		img = img / 255

		if subcoords != None:
			img,clz = extract_roi(img, clz, subcoords)

		width, height, depth = img.shape

		if batch_clz == None:
			batch_size = len(gridrefs) * (((height - patchsize) / patchstep)) * (((width - patchsize) / patchstep))
			if limit != None:
				batch_size = min(batch_size, limit)
			batch_clz = np.zeros((batch_size, len(mapping)))
			batch_img = np.zeros((batch_size, patchsize, patchsize, depth))
	
		if shuffle:
			sample_locations = shuffled_coord_generator
		else:
			sample_locations = coord_generator

		for x,y in sample_locations(width, height, patchsize, patchstep):
			patch = img[y:y+patchsize, x:x+patchsize, :]
			theclz = clz[y+patchsize/2, x+patchsize/2]

			batch_img[batch_idx, :, :, :] = patch
			batch_clz[batch_idx, :] = 0
			batch_clz[batch_idx][mapping_keys.index(theclz)] = 1 
			batch_idx = batch_idx + 1
			if batch_idx == batch_size / len(gridrefs):
				break

	if dim_ordering == 'th':
		return batch_img.transpose(0, 3, 1, 2), batch_clz
	else:
		return batch_img, batch_clz

# for batch_img, batch_clz in generate_labelled_patches(["SU4111"], 3, patchstep=50):
#  	print batch_clz
# 	for patch, clz in batch:
# 		print patch

# print load_labelled_patches(["SU4111"], 3, subcoords=((0,0), (100,100)))
