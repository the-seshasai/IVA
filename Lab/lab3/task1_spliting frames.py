import sys

import ffmpeg

sys.path.append(r'C:\ffmpeg')

input_file = 'in.mp4'
output_pattern = 'frames/frame_%04d.jpeg'  

ffmpeg.input(input_file).output(output_pattern).run()