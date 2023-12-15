import nibabel as nib
from nibabel.viewers import OrthoSlicer3D
example_filename = '/home/shiya.xu/papers/DoDNet_1/a_DynConv/outputs/liver_0.nii_label.nii.gz'
img = nib.load(example_filename)
OrthoSlicer3D(img.dataobj).show()