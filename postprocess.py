import os
import nibabel as nib
from hparam import hparams as hp
import numpy as np
from pathlib import Path


# Load a list of nii.gz files from old_path(should be a list), then stack and save as a new file to the new_path
# For example:
# old_path = ['./Case1_1.nii.gz','./Case1_2.nii.gz'] will be saved as 'new_path + stack.nii.gz' after stacking

def stack(title, len, new_path=hp.inference_dir + '/mri'):
    if not (os.path.exists(new_path)):
        os.mkdir(new_path)

    mri = []
    for i in range(int(len)):
        try:
            nii_img = nib.load(os.path.join(hp.inference_dir, title + '_' + str(i) + hp.save_arch))
        except:
            print('no such file: ' + title + '_' + str(i) + hp.save_arch)
        nii_data = nii_img.get_fdata()
        mri.append(nii_data)

    mri = np.stack(mri, axis=-1)

    affine = nii_img.affine.copy()
    hdr = nii_img.header.copy()

    new_nii = nib.Nifti1Image(mri, affine, hdr)
    nib.save(new_nii, os.path.join(new_path, title + hp.save_arch))
    print('Saved to ' + new_path + title + hp.save_arch)


if __name__ == '__main__':
    if not (os.path.exists(hp.inference_dir)):
        print('please inference first')

    inference_dir = Path(hp.inference_dir)
    paths = sorted(inference_dir.glob(hp.fold_arch))

    title = []
    for path in paths:
        if not path.name.split('.')[0].split('_')[0] in title:
            title.append(path.name.split('.')[0].split('_')[0])

    for i in title:
        stack(i, len(paths) / 3)
