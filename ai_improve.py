import ffmpeg

input_file = 'aks.mov'
output_file = 'enhanced.mp4'

ffmpeg.input(input_file).output(
    output_file,
    vf='hqdn3d,unsharp=5:5:1.0:5:5:0.0',  # Denoise + Sharpen
    vcodec='libx264',
    crf=18,  # Lower = better quality (range: 18–28)
    preset='slow',
    acodec='aac'
).run()
