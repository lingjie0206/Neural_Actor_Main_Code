import os
import argparse
import glob
import os.path
import zipfile

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--subject", required=True, help="the path of the data")
ap.add_argument("-o", "--output",  default=None,  help="output folder")
ap.add_argument("-c", "--cameras", default=None, type=str, help="start,end: 0,10")
ap.add_argument("-f", "--frames",  default=None, type=str, help="start,end: 0,50")
args = ap.parse_args()

if args.output is None:
	args.output = args.subject
if args.frames is not None:
	f_start, f_end = [int(a) for a in args.frames.split(',')]

####################### Uncompress transform files ######################
if os.path.exists(args.subject + '/transform.zip'):
	os.makedirs(args.output + '/transform', exist_ok=True)
	zip_file = zipfile.ZipFile(args.subject + '/transform.zip', 'r')
	zipinfos = [z for z in zip_file.infolist() if z.filename.endswith('.json')]

	for f in range(f_start, f_end):
		zipinfo = zipinfos[f]
		zipinfo.filename = zipinfo.filename.split('/')[-1]
		zip_file.extract(zipinfo, args.output + '/transform/')

####################### Uncompress texture video ########################
if os.path.exists(args.subject + '/tex_modifysmpluv0.1_smooth3e-2.avi'):
	os.makedirs(args.output + '/tex', exist_ok=True)
	command = "ffmpeg -nostdin -i {} -qscale:v 2".format(args.subject + '/tex_modifysmpluv0.1_smooth3e-2.avi')
	if args.frames is not None:
		command += ' -start_number {} -vframes {}'.format(f_start, f_end - f_start)
	command += " {}".format(args.output + '/tex/frame_%06d.png')
	print(command)
	os.system(command)
  
# ####################### Uncompress normal video ########################
if os.path.exists(args.subject + '/normal_modifysmpluv0.1_smooth3e-2.avi'):
	os.makedirs(args.output + '/normal', exist_ok=True)
	command = "ffmpeg -nostdin -i {} -qscale:v 2".format(args.subject + '/normal_modifysmpluv0.1_smooth3e-2.avi')
	if args.frames is not None:
		command += ' -start_number {} -vframes {}'.format(f_start, f_end - f_start)
	command += " {}".format(args.output + '/normal/frame_%06d.png')
	print(command)
	os.system(command)

# ####################### Uncompress rgb video ########################
if os.path.exists(args.subject + '/rgb_video'):
	rgb_videos = sorted(glob.glob(args.subject + '/rgb_video/*.avi'))
	if args.cameras is None:
		c_start, c_end = 0, len(rgb_videos)
	else:
		c_start, c_end = [int(a) for a in args.cameras.split(',')]

os.makedirs(args.output + '/rgb', exist_ok=True)
for f in range(f_start, f_end):
	frame = '{:06d}'.format(f)
	os.makedirs(args.output+'/rgb/'+frame, exist_ok=True)
for c in range(c_start, c_end):
	cam = '{:03d}'.format(c)
	command = "ffmpeg -nostdin -i {} -qscale:v 2".format(args.subject + '/rgb_video/' + cam + '.avi')
	if args.frames is not None:
		command += ' -start_number {} -vframes {}'.format(f_start, f_end - f_start)
	command += ' {}'.format(args.output + '/rgb/%6d/image_c_' + cam + '.png')
	print(command)
	os.system(command)

print('done')  
