import os

folder_path = '/home/tiger/dpm-solver-main/examples/ddpm_and_guided-diffusion/experiments/imagenet128_guided/neuralode_50_time_uniform_scale8.0/image_samples/images'  # Replace with the path to your folder
image_names = []

for filename in os.listdir(folder_path):
    if filename.endswith(('.jpg', '.jpeg', '.png', '.gif')):  # Add more extensions if needed
        image_names.append(filename)

attack_num = 0
total = 0

sorted_images = {}
n = len(image_names)

for name in image_names:
    temp = name.split('_')
    key = temp[0]+'_'+temp[1]
    if key not in sorted_images:
        if 'pre.png' in temp:
            if int(temp[2]) == int(temp[3]):
                total += 1
                sorted_images[key]=[int(temp[2])]
            else:
                sorted_images[key]=[int(temp[2]), int(temp[3])]
        if 'after.png' in temp:
            sorted_images[key]=[int(temp[2])]
    else:
        if 'pre.png' in temp:
            if int(temp[2]) == int(temp[3]):
                total += 1
                sorted_images[key].append(int(temp[2]))
            else:
                del sorted_images[key]
        if 'after.png' in temp:
            if len(sorted_images[key])==1:
                sorted_images[key].append(int(temp[2]))
            else:
                del sorted_images[key]

for key, value in sorted_images.items():
    if len(value)==2:
        if value[0]!=value[1]:
            attack_num+=1

print(attack_num, total)