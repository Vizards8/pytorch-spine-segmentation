import os

# Case100_1.nii.gz Case100_1_0000.nii.gz
res = ''
train = os.listdir('./Task058_Spine/imagesTr')
for i in train:
    tmp = 'mv ' + i + ' ' + i[:-7] + '_0000.nii.gz'
    tmp += '\n'
    res += tmp
with open('./imagesTr.sh', 'w') as f:
    f.write(res)

# Case100_1.nii.gz Case100_1_0000.nii.gz
res = ''
test = os.listdir('./Task058_Spine/imagesTs')
for i in test:
    tmp = 'mv ' + i + ' ' + i[:-7] + '_0000.nii.gz'
    tmp += '\n'
    res += tmp
with open('./imagesTs.sh', 'w') as f:
    f.write(res)

# mask_case100_1.nii.gz Case100_1_0000.nii.gz
res = ''
label = os.listdir('./Task058_Spine/labelsTr')
for i in label:
    tmp = 'mv ' + i + ' C' + i[6:]
    tmp += '\n'
    res += tmp
with open('./labelsTr.sh', 'w') as f:
    f.write(res)
