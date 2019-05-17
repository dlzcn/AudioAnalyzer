# Used successfully in Python2.7.10 with: 
# numpy, scipy, skimage >= 0.11.2, matplotlib >= 1.3.x and PyQt4 (and Qt 4)
import os, sys
from cx_Freeze import setup, hooks, Executable

# fixed the bug about scipy, not needed for Python XY distribution

#==============================================================================
def load_scipy_patched(finder, module):
    """the scipy module loads items within itself in a way that causes
        problems without the entire package and a number of other subpackages
        being present."""
    finder.IncludePackage("scipy._lib")  # Changed include from scipy.lib to scipy._lib
    finder.IncludePackage("scipy.misc")
hooks.load_scipy = load_scipy_patched
#==============================================================================

import Audio_Analyzer as app
# Remove the build folder, a bit slower but ensures that build contains
# the latest
import shutil
app_dir = 'Audio_Analyzer'
shutil.rmtree(app_dir, ignore_errors=True)

def move_files(src, dst, ignore=None):
    names = [f for f in os.listdir(src) \
                if os.path.isfile(os.path.join(src, f))]
    if ignore is not None:
        ignored_names = ignore(src, names)
    else:
        ignored_names = set()

    try:
        os.makedirs(dst)
    except:
        pass
    errors = []
    for name in names:
        if name in ignored_names:
            continue
        srcname = os.path.join(src, name)
        dstname = os.path.join(dst, name)
        try:
            shutil.move(srcname, dstname)
        except (IOError, os.error) as why:
            errors.append((srcname, dstname, str(why)))
        # catch the Error from the recursive copytree so that we can
        # continue with other files
        except shutil.Error as err:
            errors.extend(err.args[0])

# We need to exclude matplotlib backends not being used by this executable.  You may find
# that you need different excludes to create a working executable with your chosen backend.
# We also need to include include various numerix libraries that the other functions call.

includes = ['atexit', 'guidata', 'guiqwt', 'pyaudio', 'numpy', 
            'scipy.signal', 'scipy.fftpack', 'scipy.integrate',
            'scipy.integrate.vode', 'scipy.integrate.lsoda',
            'scipy.integrate._odepack', 'scipy.integrate._quadpack', 'scipy.integrate._dop',
            'scipy.interpolate._fitpack', 'scipy.interpolate.dfitpack',
            'scipy.interpolate._bspl', 'scipy.interpolate._interpolate','scipy.interpolate._ppoly', 
            'scipy.special._ufuncs_cxx', 'scipy.special.specfun',
            'scipy.sparse._csparsetools', 'scipy.sparse.csgraph._validation',
            'scipy.sparse.linalg.isolve._iterative',
            'scipy.sparse.linalg.eigen.arpack._arpack',
            'scipy.optimize._minpack', 'scipy.optimize._zeros',
            'scipy.fftpack._fftpack', 'scipy.fftpack.convolve',
            'scipy.ndimage._ni_support', 'scipy.ndimage._nd_image', 'scipy.ndimage._ni_label',
            'scipy.stats.statlib',
            'numpy.linalg._umath_linalg', 'numpy.core._methods', 'numpy.lib.format'
            ]
excludes = ['_tkagg', '_fltkagg', 'tcl', 'Tkconstants', 'Tkinter', 
            'pywin.debugger', 'pywin.debugger.dbgcon', 'pywin.dialogs',
            'pydoc', 'doctest', 'unittest2', 'test', 'sqlite3', 'tornado', 
            'IPython', 'OpenGL', 'OpenGL_accelerate', 'Crypto', 'matplotlib', 
            'wxPython', 'wx._controls_', 'wx._core_', 'wx._gdi_', 'wx._misc_', 'wx._windows_',
            'pandas', 'pygments', 'sphinx', 'sphinx_rtd_theme', 'spyderlib', 'spyderplugins',
            'collections.sys', 'collections._weakref']
packages = []
dll_excludes = ['tcl85.dll', 'tk85.dll']

icon_resources = 'icons\Audio_Analyzer.ico'
bitmap_resources = []
other_resources = []

data_files = []
# add images from module guidata, guiqwt
from guidata import guidata
_GUIDATADIR = os.path.dirname(guidata.__file__)
data_files += [(os.path.join(_GUIDATADIR, 'images'), 'guidata\images')]
from guiqwt import plot
_GUIQWTDIR = os.path.dirname(plot.__file__)
data_files += [(os.path.join(_GUIQWTDIR, 'images'), 'guiqwt\images')]
# add icons
data_files += [('icons\Audio_Analyzer.ico','icons\Audio_Analyzer.ico')]

options = {
    'build_exe' : {
        'build_exe': app_dir,
        'optimize': 2,
        'includes': includes,
        'excludes': excludes,
        'packages': packages,
        'bin_excludes': dll_excludes,
        'include_files': data_files,
        'include_msvcr':True,
        'zip_include_packages': ['*'],
        'zip_exclude_packages': [],    
        'silent': True
    }
}

base = None
if sys.platform == 'win32':
    base = 'Win32GUI'
    #base = 'Console'
script = 'ConsoleSetLibPathX'
executables = [
    Executable('Audio_Analyzer.py', initScript=script, base=base, icon=icon_resources)
]

setup(
    # author = 'Haifeng CHEN',
    # bug in cxFreeze, author is company_name    
    description = 'Audio Analyzer for BZZ product (freq. & G value)',
    #company_name = 'Saint-Gobain Research (Shanghai) Co. Ltd.',
    #copyright = '(C)2016 Saint-Gobain Research (Shanghai) Co., Ltd.',
    name = 'Audio_Analyzer',
    version = '1.0.%d.0' % app.REV,
    options = options,
    executables = executables
)

# move files
lib_path = os.path.join(app_dir, 'lib')
ignore = shutil.ignore_patterns('*.exe', '*.ini', '*.zip', '*.manifest', 
                                'MSVCR*', 'python[0-9][0-9].dll')

move_files(app_dir, lib_path, ignore=ignore)