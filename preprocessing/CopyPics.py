import os
import re
import shutil 

if __name__ == "__main__":
	#path = '/Users/catherinelee/Documents/DataSci/Capstone/Age/CFD Version 2.0.2/CFD 2.0.2 Images/'
	path = '/Users/catherinelee/Documents/DataSci/Capstone/Age/SCFace/SCface_database/mugshot_frontal_cropped_all/'
	target = '/Users/catherinelee/Documents/DataSci/Capstone/Age/Pics/SCFace/'

	if not os.path.exists(target):
		os.makedirs(target)
	dirlist = os.listdir(path)
	#for folder in dirlist:
		#if os.path.isdir(path+folder):
	#filelist = os.listdir(path+folder)
	for f in dirlist:
		match_obj = re.match(r'(.*)_frontal.JPG', f)
		if match_obj:
			fname = 'SCF-SUBJ-'+match_obj.group(1)+'-1-N.jpg'
			dest = os.path.join(target, fname)
			shutil.copyfile(path+f, dest)