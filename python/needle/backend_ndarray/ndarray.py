import operator
import math
from functools import reduce
import numpy as np
from . import ndarray_backend_numpy
from . import ndarray_backend_cpu


# math.prod not in Python 3.7
def prod(x):
    return reduce(operator.mul, x, 1)


class BackendDevice:
    """A backend device, wraps the implementation module."""

    def __init__(self, name, mod):
        self.name = name
        self.mod = mod

    def __eq__(self, other):
        return self.name == other.name

    def __repr__(self):
        return self.name + "()"

    def __getattr__(self, name):
        return getattr(self.mod, name)

    def enabled(self):
        return self.mod is not None

    def randn(self, *shape, dtype="float32"):
        # note: numpy doesn't support types within standard random routines, and
        # .astype("float32") does work if we're generating a singleton
        return NDArray(numpy.random.randn(*shape).astype(dtype), device=self)

    def rand(self, *shape, dtype="float32"):
        # note: numpy doesn't support types within standard random routines, and
        # .astype("float32") does work if we're generating a singleton
        return NDArray(numpy.random.rand(*shape).astype(dtype), device=self)

    def one_hot(self, n, i, dtype="float32"):
        return NDArray(numpy.eye(n, dtype=dtype)[i], device=self)

    def empty(self, shape, dtype="float32"):
        dtype = "float32" if dtype is None else dtype
        assert dtype == "float32"
        return NDArray.make(shape, device=self)

    def full(self, shape, fill_value, dtype="float32"):
        dtype = "float32" if dtype is None else dtype
        assert dtype == "float32"
        arr = self.empty(shape, dtype)
        arr.fill(fill_value)
        return arr


def cuda():
    """Return cuda device"""
    try:
        from . import ndarray_backend_cuda

        return BackendDevice("cuda", ndarray_backend_cuda)
    except ImportError:
        return BackendDevice("cuda", None)


def cpu_numpy():
    """Return numpy device"""
    return BackendDevice("cpu_numpy", ndarray_backend_numpy)


def cpu():
    """Return cpu device"""
    return BackendDevice("cpu", ndarray_backend_cpu)


def default_device():
    return cpu_numpy()


def all_devices():
    """return a list of all available devices"""
    return [cpu(), cuda(), cpu_numpy()]


class NDArray:
    """A generic ND array class that may contain multiple different backends
    i.e., a Numpy backend, a native CPU backend, or a GPU backend.
    This class will only contains those functions that you need to implement
    to actually get the desired functionality for the programming examples
    in the homework, and no more.
    For now, for simplicity the class only supports float32 types, though
    this can be extended if desired.
    """

    def __init__(self, other, device=None):
        """ Create by copying another NDArray, or from numpy """
        if isinstance(other, NDArray):
            # create a copy of existing NDArray
            if device is None:
                device = other.device
            self._init(other.to(device) + 0.0)  # this creates a copy
        elif isinstance(other, np.ndarray):
            # create copy from numpy array
            device = device if device is not None else default_device()
            array = self.make(other.shape, device=device)
            array.device.from_numpy(np.ascontiguousarray(other), array._handle)
            self._init(array)
        else:
            # see if we can create a numpy array from input
            array = NDArray(np.array(other), device=device)
            self._init(array)

    def _init(self, other):
        self._shape = other._shape
        self._strides = other._strides
        self._offset = other._offset
        self._device = other._device
        self._handle = other._handle

    @staticmethod
    def compact_strides(shape):
        """ Utility function to compute compact strides """
        stride = 1
        res = []
        for i in range(1, len(shape) + 1):
            res.append(stride)
            stride *= shape[-i]
        return tuple(res[::-1])

    @staticmethod
    def make(shape, strides=None, device=None, handle=None, offset=0):
        """Create a new NDArray with the given properties.  This will allocation the
        memory if handle=None, otherwise it will use the handle of an existing
        array."""
        array = NDArray.__new__(NDArray)
        array._shape = tuple(shape)
        array._strides = NDArray.compact_strides(shape) if strides is None else strides
        array._offset = offset
        array._device = device if device is not None else default_device()
        if handle is None:
            array._handle = array.device.Array(prod(shape))
        else:
            array._handle = handle
        return array

    ### Properies and string representations
    @property
    def shape(self):
        return self._shape

    @property
    def strides(self):
        return self._strides

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        # only support float32 for now
        return "float32"

    @property
    def ndim(self):
        """ Return number of dimensions. """
        return len(self._shape)

    @property
    def size(self):
        return prod(self._shape)

    def __repr__(self):
        return "NDArray(" + self.numpy().__str__() + f", device={self.device})"

    def __str__(self):
        return self.numpy().__str__()

    ### Basic array manipulation
    def fill(self, value):
        """ Fill (in place) with a constant value. """
        self._device.fill(self._handle, value)

    def to(self, device):
        """ Convert between devices, using to/from numpy calls as the unifying bridge. """
        if device == self.device:
            return self
        else:
            return NDArray(self.numpy(), device=device)

    def numpy(self):
        """ convert to a numpy array """
        return self.device.to_numpy(
            self._handle, self.shape, self.strides, self._offset
        )

    def is_compact(self):
        """Return true if array is compact in memory and internal size equals product
        of the shape dimensions"""
        return (
                self._strides == self.compact_strides(self._shape)
                and prod(self.shape) == self._handle.size
        )

    def compact(self):
        """ Convert a matrix to be compact """
        if self.is_compact():
            return self
        else:
            out = NDArray.make(self.shape, device=self.device)
            self.device.compact(
                self._handle, out._handle, self.shape, self.strides, self._offset
            )
            return out

    def as_strided(self, shape, strides):
        """ Restride the matrix without copying memory. """
        assert len(shape) == len(strides)
        return NDArray.make(
            shape, strides=strides, device=self.device, handle=self._handle
        )

    @property
    def flat(self):
        return self.reshape((self.size,))

    @staticmethod
    def calculate_stride_by_shape(shape):
        strides = [1, ]
        i = len(shape) - 2
        while i >= 0:
            strides.append(strides[-1] * shape[i + 1])
            i -= 1
        strides = tuple(strides[::-1])
        return strides

    def split(self, axis):
        new_shape = list(self._shape)
        new_shape.pop(axis)
        
        slices = []
        for i, d in enumerate(self._shape):
            if i != axis:
                slices.append(slice(0, d, 1))
            else:
                slices.append(0)
        
        splits = []
        for i in range(self._shape[axis]):
            index_to_set = list(slices)
            index_to_set[axis] = i
            cur_split = self[tuple(index_to_set)]

            splits.append(cur_split.reshape(tuple(new_shape)))
        
        
        return tuple(splits)
        

    def stack(self, array_list, axis):
        new_shape = list(self._shape)
        new_shape.insert(axis, 1)
        
        compact_list = []
        for a in array_list:
            compact_list.append(a.reshape(tuple(new_shape)).compact())
        
        stack_shape = list(compact_list[0]._shape)
        stack_shape[axis] *= len(compact_list)
        
        new_strides = self.calculate_stride_by_shape(stack_shape)
        
        stacked_arr = self.make(
            shape=tuple(stack_shape),
            strides=tuple(new_strides),
            device=self.device,
        )
        
        slices = []
        for i, d in enumerate(stack_shape):
            if i != axis:
                slices.append(slice(0, d, 1))
            else:
                slices.append(0)
        
        for i, arr in enumerate(compact_list):
            index_to_set = list(slices)
            index_to_set[axis] = i
            stacked_arr[tuple(index_to_set)] = arr
        
        return stacked_arr
        

    def concat(self, array_list, axis):
        pass
        # TODO: concat!
        # compact_list = []
        # for a in array_list:
        #     compact_list.append(a.compact())
            
        # first_arr = compact_list[0]
        # new_shape = list(first_arr._shape)
        # new_shape[axis] *= len(compact_list)
        # new_shape = tuple(new_shape)
        
        # handle = first_arr._handle
        # for arr in compact_list[1:]:
        #     handle = handle + a._handle 
        # new_strides = self.calculate_stride_by_shape(new_shape)
        
        # print(new_shape, new_strides, handle)
        # return self.make(
        #     shape=new_shape,
        #     strides=new_strides,
        #     device=first_arr.device,
        #     handle=handle,
        #     offset=first_arr._offset
        # )

    def reshape(self, new_shape):
        """
        Reshape the matrix without copying memory.  This will return a matrix
        that corresponds to a reshaped array but points to the same memory as
        the original array.
        Raises:
            ValueError if product of current shape is not equal to the product
            of the new shape, or if the matrix is not compact.
        Args:
            new_shape (tuple): new shape of the array
        Returns:
            NDArray : reshaped array; this will point to the same memory as the original NDArray.
        """

        new_strides = self.calculate_stride_by_shape(new_shape)
        return self.make(
            shape=new_shape,
            strides=new_strides,
            device=self.device,
            handle=self._handle,
            offset=self._offset
        )

    def permute(self, new_axes):
        """
        Permute order of the dimensions.  new_axes describes a permutation of the
        existing axes, so e.g.:
          - If we have an array with dimension "BHWC" then .permute((0,3,1,2))
            would convert this to "BCHW" order.
          - For a 2D array, .permute((1,0)) would transpose the array.
        Like reshape, this operation should not copy memory, but achieves the
        permuting by just adjusting the shape/strides of the array.  That is,
        it returns a new array that has the dimensions permuted as desired, but
        which points to the same memory as the original array.
        Args:
            new_axes (tuple): permutation order of the dimensions
        Returns:
            NDarray : new NDArray object with permuted dimensions, pointing
            to the same memory as the original NDArray (i.e., just shape and
            strides changed).
        """

        new_shape = tuple([self.shape[i] for i in new_axes])
        new_strides = tuple([self.strides[i] for i in new_axes])

        return self.make(
            shape=new_shape,
            strides=new_strides,
            device=self.device,
            handle=self._handle,
            offset=self._offset
        )

    def broadcast_to(self, new_shape):
        """
        Broadcast an array to a new shape.  new_shape's elements must be the
        same as the original shape, except for dimensions in the self where
        the size = 1 (which can then be broadcast to any size).  As with the
        previous calls, this will not copy memory, and just achieves
        broadcasting by manipulating the strides.
        Raises:
            assertion error if new_shape[i] != shape[i] for all i where
            shape[i] != 1
        Args:
            new_shape (tuple): shape to broadcast to
        Returns:
            NDArray: the new NDArray object with the new broadcast shape; should
            point to the same memory as the original array.
        """

        for i, d in enumerate(new_shape):
            if self.shape[i] != 1:
                assert self.shape[i] == new_shape[i]

        new_strides = tuple([s if d > 1 else 0 for s, d in zip(self.strides, self.shape)])

        return self.make(
            shape=new_shape,
            strides=new_strides,
            device=self.device,
            handle=self._handle,
            offset=self._offset
        )

    def process_slice(self, sl, dim):
        """ Convert a slice to an explicit start/stop/step """
        start, stop, step = sl.start, sl.stop, sl.step
        if start == None:
            start = 0
        if start < 0:
            start = self.shape[dim]
        if stop == None:
            stop = self.shape[dim]
        if stop < 0:
            stop = self.shape[dim] + stop
        if step == None:
            step = 1

        # we're not gonna handle negative strides and that kind of thing
        assert stop > start, "Start must be less than stop"
        assert step > 0, "No support for  negative increments"
        return slice(start, stop, step)

    def __getitem__(self, idxs):
        """
        The __getitem__ operator in Python allows us to access elements of our
        array.  When passed notation such as a[1:5,:-1:2,4,:] etc, Python will
        convert this to a tuple of slices and integers (for singletons like the
        '4' in this example).  Slices can be a bit odd to work with (they have
        three elements .start .stop .step), which can be None or have negative
        entries, so for simplicity we wrote the code for you to convert these
        to always be a tuple of slices, one of each dimension.
        For this tuple of slices, return an array that subsets the desired
        elements.  As before, this can be done entirely through compute a new
        shape, stride, and offset for the new "view" into the original array,
        pointing to the same memory
        Raises:
            AssertionError if a slice has negative size or step, or if number
            of slices is not equal to the number of dimension (the stub code
            already raises all these errors.
        Args:
            idxs tuple: (after stub code processes), a tuple of slice elements
            corresponding to the subset of the matrix to get
        Returns:
            NDArray: a new NDArray object corresponding to the selected
            subset of elements.  As before, this should not copy memory but just
            manipulate the shape/strides/offset of the new array, referencing
            the same array as the original one.
        """

        # handle singleton as tuple, everything as slices
        if not isinstance(idxs, tuple):
            idxs = (idxs,)
        idxs = tuple(
            [
                self.process_slice(s, i) if isinstance(s, slice) else slice(s, s + 1, 1)
                for i, s in enumerate(idxs)
            ]
        )
        assert len(idxs) == self.ndim, "Need indexes equal to number of dimensions"

        new_shape = []
        for i, (d, s) in enumerate(zip(self.shape, idxs)):
            new_d = math.ceil((min(s.stop, d) - max(0, s.start)) / s.step)
            new_shape.append(new_d)

        offset = 0
        new_strides = []
        for stride, sl in zip(self.strides, idxs):
            offset += stride * sl.start
            new_strides.append(stride if sl.step == 1 else stride * sl.step)

        return self.make(
            shape=new_shape,
            strides=tuple(new_strides),
            device=self.device,
            handle=self._handle,
            offset=offset
        )

    def __setitem__(self, idxs, other):
        """Set the values of a view into an array, using the same semantics
        as __getitem__()."""
        view = self.__getitem__(idxs)
        if isinstance(other, NDArray):
            assert prod(view.shape) == prod(other.shape)
            self.device.ewise_setitem(
                other.compact()._handle,
                view._handle,
                view.shape,
                view.strides,
                view._offset,
            )
        else:
            self.device.scalar_setitem(
                prod(view.shape),
                other,
                view._handle,
                view.shape,
                view.strides,
                view._offset,
            )

    ### Collection of elementwise and scalar function: add, multiply, boolean, etc

    def ewise_or_scalar(self, other, ewise_func, scalar_func):
        """Run either an element-wise or scalar version of a function,
        depending on whether "other" is an NDArray or scalar
        """
        out = NDArray.make(self.shape, device=self.device)
        if isinstance(other, NDArray):
            assert self.shape == other.shape, "operation needs two equal-sized arrays"
            ewise_func(self.compact()._handle, other.compact()._handle, out._handle)
        else:
            scalar_func(self.compact()._handle, other, out._handle)
        return out

    def __add__(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_add, self.device.scalar_add
        )

    __radd__ = __add__

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_mul, self.device.scalar_mul
        )

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_div, self.device.scalar_div
        )

    def __neg__(self):
        return self * (-1)

    def __pow__(self, other):
        out = NDArray.make(self.shape, device=self.device)
        self.device.scalar_power(self.compact()._handle, other, out._handle)
        return out

    def maximum(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_maximum, self.device.scalar_maximum
        )

    ### Binary operators all return (0.0, 1.0) floating point values, could of course be optimized
    def __eq__(self, other):
        return self.ewise_or_scalar(other, self.device.ewise_eq, self.device.scalar_eq)

    def __ge__(self, other):
        return self.ewise_or_scalar(other, self.device.ewise_ge, self.device.scalar_ge)

    def __ne__(self, other):
        return 1 - (self == other)

    def __gt__(self, other):
        return (self >= other) * (self != other)

    def __lt__(self, other):
        return 1 - (self >= other)

    def __le__(self, other):
        return 1 - (self > other)

    ### Elementwise functions

    def log(self):
        out = NDArray.make(self.shape, device=self.device)
        self.device.ewise_log(self.compact()._handle, out._handle)
        return out

    def exp(self):
        out = NDArray.make(self.shape, device=self.device)
        self.device.ewise_exp(self.compact()._handle, out._handle)
        return out

    def tanh(self):
        out = NDArray.make(self.shape, device=self.device)
        self.device.ewise_tanh(self.compact()._handle, out._handle)
        return out

        ### Matrix multiplication
    def __matmul__(self, other):
        """Matrix multplication of two arrays.  This requires that both arrays
        be 2D (i.e., we don't handle batch matrix multiplication), and that the
        sizes match up properly for matrix multiplication.

        In the case of the CPU backend, you will implement an efficient "tiled"
        version of matrix multiplication for the case when all dimensions of
        the array are divisible by self.device.__tile_size__.  In this case,
        the code below will restride and compact the matrix into tiled form,
        and then pass to the relevant CPU backend.  For the CPU version we will
        just fall back to the naive CPU implementation if the array shape is not
        a multiple of the tile size

        The GPU (and numpy) versions don't have any tiled version (or rather,
        the GPU version will just work natively by tiling any input size).
        """

        assert self.ndim == 2 and other.ndim == 2
        assert self.shape[1] == other.shape[0]

        m, n, p = self.shape[0], self.shape[1], other.shape[1]

        # if the matrix is aligned, use tiled matrix multiplication
        if hasattr(self.device, "matmul_tiled") and all(
            d % self.device.__tile_size__ == 0 for d in (m, n, p)
        ):

            def tile(a, tile):
                return a.as_strided(
                    (a.shape[0] // tile, a.shape[1] // tile, tile, tile),
                    (a.shape[1] * tile, tile, a.shape[1], 1),
                )

            t = self.device.__tile_size__
            a = tile(self.compact(), t).compact()
            b = tile(other.compact(), t).compact()
            out = NDArray.make((a.shape[0], b.shape[1], t, t), device=self.device)
            self.device.matmul_tiled(a._handle, b._handle, out._handle, m, n, p)

            return (
                out.permute((0, 2, 1, 3))
                .compact()
                .reshape((self.shape[0], other.shape[1]))
            )

        else:
            out = NDArray.make((m, p), device=self.device)
            self.device.matmul(
                self.compact()._handle, other.compact()._handle, out._handle, m, n, p
            )
            return out

    ### Reductions, i.e., sum/max over all element or over given axis
    def reduce_view_out(self, axis):
        """ Return a view to the array set up for reduction functions and output array. """
        if axis is None:
            view = self.reshape((1,) * (self.ndim - 1) + (prod(self.shape),))
            out = NDArray.make((1,) * self.ndim, device=self.device)
        else:
            view = self.permute(
                tuple([a for a in range(self.ndim) if a != axis]) + (axis,)
            )
            out = NDArray.make(
                tuple([1 if i == axis else s for i, s in enumerate(self.shape)]),
                device=self.device,
            )
        return view, out

    def sum(self, axis=None, keepdims=False):

        begin_shape = self._shape
        view, out = self.reduce_view_out(axis)
        self.device.reduce_sum(view.compact()._handle, out._handle, view.shape[-1])
        if not keepdims:
            if axis != None:
                squeezed_shape = tuple([d for i,d in enumerate(begin_shape) if i != axis])
                out = out.reshape(squeezed_shape)
            else:
                out = out.reshape((1,))
        return out
        
    def dilate(self, dilation, axes):
        shape = list(self.shape)
        # axes = (ax for ax in axes if ax < len(shape))
        for ax in axes:
            if ax >= len(shape):
                return self
        
        for ax in axes:
            shape[ax] *= (dilation + 1)
            
        strides = self.calculate_stride_by_shape(shape)
        
        slices = []
        for i, d in enumerate(shape):
            if i in axes:
                slices.append(slice(None, None, dilation+1))
            else:
                slices.append(slice(None, None, None))
                
        arr = self.make(
            shape=shape,
            strides=strides,
            device=self.device,
        )
        
        arr.fill(0.0)
        arr[tuple(slices)] = self
        return arr
        
    def undilate(self, dilation, axes):
        
        
        strides = list(self._strides)
        shape = list(self.shape)
        
        for ax in axes:
            if ax >= len(shape):
                return self
        
        for ax in axes:
            shape[ax] = int(shape[ax] /(dilation + 1))
            
        for ax in axes:
            strides[ax] *= (dilation + 1)
        
        
        arr = self.make(
            shape=shape,
            strides=strides,
            device=self.device,
            handle=self._handle,
            offset=self._offset
        )
        
        
        return arr
        
        
    # def conv_im2col(Z, weight):
    #     N,H,W,C_in = Z.shape
    #     K,_,_,C_out = weight.shape
    #     Ns, Hs, Ws, Cs = Z.strides
        
    #     inner_dim = K * K * C_in
    #     A = np.lib.stride_tricks.as_strided(Z, shape = (N, H-K+1, W-K+1, K, K, C_in),
    #                                         strides = (Ns, Hs, Ws, Hs, Ws, Cs)).reshape(-1,inner_dim)
    #     out = A @ weight.reshape(-1, C_out)
    #     return out.reshape(N,H-K+1,W-K+1,C_out)
        
    def conv2d(self, w, stride=1, padding=0):
        pad_axes = ((0, 0), (padding, padding), (padding, padding), (0, 0))
        arr = self.pad(pad_axes)
        
        N, H, W, C_in = arr.shape
        K, _, _, C_out = w.shape
        Ns, Hs, Ws, Cs = arr._strides
        
        inner_dim = K * K * C_in
        #print(self.shape, self)
        #print(w.shape)
        s = int(((H-K) / stride) + 1)
        A = self.make(
            shape=(N, s, s, K, K, C_in),
            strides=(Ns, Hs*stride, Ws*stride, Hs, Ws, Cs),
            device=arr.device,
            handle=arr._handle,
            offset=arr._offset
        ).compact()
        
        # print(A.shape)
        
        A = A.reshape((prod(A.shape)//inner_dim, inner_dim))
        w = w.reshape((prod(w.shape)//C_out, C_out))
        
        out = matmul(A, w)
        out = out.reshape((N, s, s, C_out))
        
        #print(out.shape, out)
        return out
        
        
        
    def flip(self, axis):
        shape = list(self.shape)
        strides = list(self._strides)
        
        for ax in axis:
            strides[ax] = -strides[ax]
        
        prods = []
        cum = 1
        for d in shape[::-1]:
            cum *= d
            prods.append(cum)
        
        prods = prods[::-1]
        ax_offsets = []
        for i in range(len(prods)-1):
            ax_offsets.append(prods[i]-prods[i+1])
        ax_offsets.append(prods[-1]-1)
        
        offset = 0
        for ax in axis:
            offset += ax_offsets[ax]
            
        arr = self.make(
            shape=self.shape,
            strides=strides,
            device=self.device,
            handle=self._handle,
            offset=offset
        )
        
        arr.compact()
        
        return arr


    def max(self, axis=None, keepdims=False):
        begin_shape = self._shape
        view, out = self.reduce_view_out(axis)
        self.device.reduce_max(view.compact()._handle, out._handle, view.shape[-1])
        if not keepdims:
            squeezed_shape = tuple([d for i,d in enumerate(begin_shape) if i != axis])
            out = out.reshape(squeezed_shape)
        
        return out
        
    def pad(self, ax_pads):
        padded_shape = list(self.shape)
        for i, d in enumerate(ax_pads):
            padded_shape[i] += d[0] + d[1]
            
        padded_data = self.make(
            shape=padded_shape,
            device=self.device,
        )
        
        padded_data.fill(0.0)
        
        slices = []
        for ax, d in zip(ax_pads, padded_shape):
            slices.append(slice(ax[0], d-ax[1], 1))
        
        padded_data[tuple(slices)] = self
        return padded_data
        
        
def array(a, dtype="float32", device=None):
    """ Convenience methods to match numpy a bit more closely."""
    dtype = "float32" if dtype is None else dtype
    assert dtype == "float32"
    return NDArray(a, device=device)


def empty(shape, dtype="float32", device=None):
    device = device if device is not None else default_device()
    return devie.empty(shape, dtype)


def full(shape, fill_value, dtype="float32", device=None):
    device = device if device is not None else default_device()
    return device.full(shape, fill_value, dtype)


def broadcast_to(array, new_shape):
    return array.broadcast_to(new_shape)
    
    
def reshape(array, new_shape):
    return array.reshape(new_shape)


def maximum(a, b):
    return a.maximum(b)

def log(a):
    return a.log()


def exp(a):
    return a.exp()

def matmul(a, b):
    return a.__matmul__(b)
    
def transpose(a, axes):
    return a.permute(axes)

def tanh(a):
    return a.tanh()

def flip(a, axes):
    return a.flip(axes)

def summation(a, axis=None, keepdims=False):
    return a.sum(axis=axis, keepdims=keepdims)

def sum(a, axis=None, keepdims=False):
    return a.sum(axis=axis, keepdims=keepdims)
    
def reduce_max(a, axis=None, keepdims=False):
    return a.max(axis=axis, keepdims=keepdims)
    
def stack(array_list, axis):
    return array_list[0].stack(array_list, axis)
    
def split(a, axis):
    return a.split(axis)
    
def dilate(a, dilation, axes):
    return a.dilate(dilation, axes)

def undilate(a, dilation, axes):
    return a.undilate(dilation, axes)