import vtk
import numpy as np
from vtk.util import numpy_support


spa = []

def vtk_to_numpy(data):
  """
  This function is to transform vtk to numpy

  Args
      data: vtk data

  Return: numpy data
  """
  temp = numpy_support.vtk_to_numpy(data.GetPointData().GetScalars())
  dims = data.GetDimensions()
  global spa
  spa = data.GetSpacing()
  component = data.GetNumberOfScalarComponents()
  if component == 1:
    numpy_data = temp.reshape(dims[2], dims[1], dims[0])
    numpy_data = numpy_data.transpose(2,1,0)
  elif component == 3 or component == 4:
    if dims[2] == 1: # a 2D RGB image
      numpy_data = temp.reshape(dims[1], dims[0], component)
      numpy_data = numpy_data.transpose(0, 1, 2)
      numpy_data = np.flipud(numpy_data)
    else:
      raise RuntimeError('unknow type')
  return numpy_data

def numpy_to_vtk(data, multi_component=False, type='char'):
  '''
  multi_components: rgb has 3 components
  type：float or char
  '''
  if type == 'float':
    data_type = vtk.VTK_FLOAT
  elif type == 'char':
    data_type = vtk.VTK_UNSIGNED_CHAR
  else:
    raise RuntimeError('unknown type')
  if multi_component == False:
    if len(data.shape) == 2:
      data = data[:, :, np.newaxis]
    flat_data_array = data.transpose(2,1,0).flatten()
    vtk_data = numpy_support.numpy_to_vtk(num_array=flat_data_array, deep=True, array_type=data_type)
    shape = data.shape
  else:
    assert len(data.shape) == 3, 'only test for 2D RGB'
    flat_data_array = data.transpose(1, 0, 2)
    flat_data_array = np.reshape(flat_data_array, newshape=[-1, data.shape[2]])
    vtk_data = numpy_support.numpy_to_vtk(num_array=flat_data_array, deep=True, array_type=data_type)
    shape = [data.shape[0], data.shape[1], 1]
  img = vtk.vtkImageData()
  img.GetPointData().SetScalars(vtk_data)
  img.SetDimensions(shape[0], shape[1], shape[2])
  img.SetSpacing(spa[0], spa[1], spa[2])
  return img

def vtk_data_loader(data_path):
  """
  This function is to load vtk data

  Args
      data_path: vtk data path

  Return: vtk data transformed to numpy
  """
  vtk_reader = vtk.vtkXMLImageDataReader()
  vtk_reader.SetFileName(data_path)
  vtk_reader.Update()
  vtk_data = vtk_reader.GetOutput()

  npdata = vtk_to_numpy(vtk_data).astype(np.float32)

  # data = np.zeros((x, y, z))

  return npdata

def save_vtk(img, output_path):
  writer = vtk.vtkXMLImageDataWriter()
  writer.SetFileName(output_path)
  writer.SetInputData(img)
  writer.Write()

def toRGB(volume):
    rgbVolume = np.zeros((volume.shape[0], volume.shape[1], volume.shape[2], 3))
    rgbVolume[:,:,:,0] = volume
    rgbVolume[:,:,:,1] = volume
    rgbVolume[:,:,:,2] = volume

    return rgbVolume