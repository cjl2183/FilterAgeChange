def load_img(path, grayscale=False, target_size=None):
	from PIL import Image
	img = Image.open(path)
	if grayscale:
	    img = img.convert('L')
	else:  # Ensure 3 channel even when loaded image is grayscale
	    img = img.convert('RGB')
	if target_size:
	    img = img.resize((target_size, target_size))
	return img

def transform_image_RGB(path, target_size=227):
    '''
    INPUTS: Path to Image (String)
    OUTPUTS: 4D Numpy Tensor

    Opens file with cv2, transposes to RGB, adds dimension and returns
    '''
    from scipy import misc
    import numpy as np

    try:
	    im = misc.imread(path)
	    im = misc.imresize(im, (target_size, target_size)).astype(np.float32)
	    im = im.transpose((2, 0, 1))
	    return np.expand_dims(im, axis=0)
    except Exception:
	    print "Issues with file {}".format(path)
	    return None

def transform_image(path, target_size=227):
    '''
    INPUTS: Path to Image (String)
    OUTPUTS: 4D Numpy Tensor

    Opens file with cv2, transposes to RGB, adds dimension and returns
    '''
    from scipy import misc
    import os
    import numpy as np

    mean_pixel = [0, 0, 0]

#    if os.stat(path).st_size < 4000:
#        return None

    try:
	    im = misc.imread(path)
	    im = im[:,:,[2,1,0]]
	    im = misc.imresize(im, (target_size, target_size)).astype(np.float32)

	    for channel in xrange(3):
	        im[:,:,channel] -= mean_pixel[channel]
	    im = im.transpose((2, 1, 0))

	    return np.expand_dims(im, axis=0)
    except Exception:
	    print "Issues with file {}".format(path)
	    return None

def img_to_array(img, dim_ordering='th'):
	import numpy as np
	
	if dim_ordering not in ['th', 'tf']:
	    raise Exception('Unknown dim_ordering: ', dim_ordering)
	# image has dim_ordering (height, width, channel)
	x = np.asarray(img, dtype='float32')
	print x.shape
	if len(x.shape) == 3:
	    if dim_ordering == 'th':
	        x = x.transpose(2, 0, 1)
	elif len(x.shape) == 2:
	    if dim_ordering == 'th':
	        x = x.reshape((1, x.shape[0], x.shape[1]))
	    else:
	        x = x.reshape((x.shape[0], x.shape[1], 1))
	else:
	    raise Exception('Unsupported image shape: ', x.shape)
	return x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))