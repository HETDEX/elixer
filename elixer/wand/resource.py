""":mod:`wand.resource` --- Global resource management
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There is the global resource to manage in MagickWand API. This module
implements automatic global resource management through reference counting.

"""
import contextlib
import ctypes
import warnings

from .api import libmagick, library
from .compat import abc, string_type, text
from .exceptions import TYPE_MAP, WandException
from .version import MAGICK_VERSION_NUMBER

__all__ = ('genesis', 'terminus', 'increment_refcount', 'decrement_refcount',
           'limits', 'Resource', 'ResourceLimits', 'DestroyedResourceError',
           'safe_copy')


def genesis():
    """Instantiates the MagickWand API.

    .. warning::

       Don't call this function directly. Use :func:`increment_refcount()` and
       :func:`decrement_refcount()` functions instead.

    """
    library.MagickWandGenesis()


def terminus():
    """Cleans up the MagickWand API.

    .. warning::

       Don't call this function directly. Use :func:`increment_refcount()` and
       :func:`decrement_refcount()` functions instead.

    """
    library.MagickWandTerminus()


#: (:class:`numbers.Integral`) The internal integer value that maintains
#: the number of referenced objects.
#:
#: .. warning::
#:
#:    Don't touch this global variable. Use :func:`increment_refcount()` and
#:    :func:`decrement_refcount()` functions instead.
#:
reference_count = 0


def increment_refcount():
    """Increments the :data:`reference_count` and instantiates the MagickWand
    API if it is the first use.

    """
    global reference_count
    if reference_count:
        reference_count += 1
    else:
        genesis()
        reference_count = 1


def decrement_refcount():
    """Decrements the :data:`reference_count` and cleans up the MagickWand
    API if it will be no more used.

    """
    global reference_count
    if not reference_count:
        raise RuntimeError('wand.resource.reference_count is already zero')
    reference_count -= 1
    if not reference_count:
        terminus()


def safe_copy(ptr):
    """Safely cast memory address to char pointer, convert to python string,
    and immediately free resources.

    :param ptr: The memory address to convert to text string.
    :type ptr: :class:`ctypes.c_void_p`
    :returns: :class:`tuple` (:class:`ctypes.c_void_p`, :class:`str`)

    .. versionadded:: 0.5.3
    """
    string = ''
    if bool(ptr):
        string = ctypes.cast(ptr, ctypes.c_char_p).value
        ptr = library.MagickRelinquishMemory(ptr)
    return ptr, text(string)


class Resource(object):
    """Abstract base class for MagickWand object that requires resource
    management. Its all subclasses manage the resource semiautomatically
    and support :keyword:`with` statement as well::

        with Resource() as resource:
            # use the resource...
            pass

    It doesn't implement constructor by itself, so subclasses should
    implement it. Every constructor should assign the pointer of its
    resource data into :attr:`resource` attribute inside of :keyword:`with`
    :meth:`allocate()` context.  For example::

        class Pizza(Resource):
            '''My pizza yummy.'''

            def __init__(self):
                with self.allocate():
                    self.resource = library.NewPizza()

    .. versionadded:: 0.1.2

    """

    #: (:class:`ctypes.CFUNCTYPE`) The :mod:`ctypes` predicate function
    #: that returns whether the given pointer (that contains a resource data
    #: usuaully) is a valid resource.
    #:
    #: .. note::
    #:
    #:    It is an abstract attribute that has to be implemented
    #:    in the subclass.
    c_is_resource = NotImplemented

    #: (:class:`ctypes.CFUNCTYPE`) The :mod:`ctypes` function that destroys
    #: the :attr:`resource`.
    #:
    #: .. note::
    #:
    #:    It is an abstract attribute that has to be implemented
    #:    in the subclass.
    c_destroy_resource = NotImplemented

    #: (:class:`ctypes.CFUNCTYPE`) The :mod:`ctypes` function that gets
    #: an exception from the :attr:`resource`.
    #:
    #: .. note::
    #:
    #:    It is an abstract attribute that has to be implemented
    #:    in the subclass.
    c_get_exception = NotImplemented

    #: (:class:`ctypes.CFUNCTYPE`) The :mod:`ctypes` function that clears
    #: an exception of the :attr:`resource`.
    #:
    #: .. note::
    #:
    #:    It is an abstract attribute that has to be implemented
    #:    in the subclass.
    c_clear_exception = NotImplemented

    @property
    def resource(self):
        """Internal pointer to the resource instance. It may raise
        :exc:`DestroyedResourceError` when the resource has destroyed already.

        """
        if getattr(self, 'c_resource', None) is None:
            raise DestroyedResourceError(repr(self) + ' is destroyed already')
        return self.c_resource

    @resource.setter
    def resource(self, resource):
        # Delete the existing resource if there is one
        if getattr(self, 'c_resource', None):
            self.destroy()

        if self.c_is_resource(resource):
            self.c_resource = resource
        else:
            raise TypeError(repr(resource) + ' is an invalid resource')
        increment_refcount()

    @resource.deleter
    def resource(self):
        self.c_destroy_resource(self.resource)
        self.c_resource = None

    @contextlib.contextmanager
    def allocate(self):
        """Allocates the memory for the resource explicitly. Its subclasses
        should assign the created resource into :attr:`resource` attribute
        inside of this context. For example::

            with resource.allocate():
                resource.resource = library.NewResource()

        """
        increment_refcount()
        try:
            yield self
        except:  # noqa: E722
            decrement_refcount()
            raise

    def destroy(self):
        """Cleans up the resource explicitly. If you use the resource in
        :keyword:`with` statement, it was called implicitly so have not to
        call it.

        """
        del self.resource
        decrement_refcount()

    def get_exception(self):
        """Gets a current exception instance.

        :returns: a current exception. it can be ``None`` as well if any
                  errors aren't occurred
        :rtype: :class:`wand.exceptions.WandException`

        """
        severity = ctypes.c_int()
        desc = self.c_get_exception(self.resource, ctypes.byref(severity))
        if severity.value == 0:
            return
        self.c_clear_exception(self.resource)
        exc_cls = TYPE_MAP[severity.value]
        message = desc.value
        if not isinstance(message, string_type):
            message = message.decode(errors='replace')
        return exc_cls(message)

    def raise_exception(self, stacklevel=1):
        """Raises an exception or warning if it has occurred."""
        e = self.get_exception()
        if isinstance(e, Warning):
            warnings.warn(e, stacklevel=stacklevel + 1)
        elif isinstance(e, Exception):
            raise e

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.destroy()

    def __del__(self):
        try:
            self.destroy()
        except DestroyedResourceError:
            pass


class DestroyedResourceError(WandException, ReferenceError, AttributeError):
    """An error that rises when some code tries access to an already
    destroyed resource.

    .. versionchanged:: 0.3.0
       It becomes a subtype of :exc:`wand.exceptions.WandException`.

    """


class ResourceLimits(abc.MutableMapping):
    """Wrapper for MagickCore resource limits.
    Useful for dynamically reducing system resources before attempting risky,
    or slow running, :class:`~wand.image.Image` operations.

    For example::

       from wand.image import Image
       from wand.resource import limits

       # Use 100MB of ram before writing temp data to disk.
       limits['memory'] = 1024 * 1024 * 100
       # Reject images larger than 1000x1000.
       limits['width'] = 1000
       limits['height'] = 1000

       # Debug resources used.
       with Image(filename='user.jpg') as img:
           print('Using {0} of {1} memory'.format(limits.resource('memory'),
                                                  limits['memory']))

       # Dump list of all limits.
       for label in limits:
           print('{0} => {1}'.format(label, limits[label]))

    Available resource keys:

    - ``'area'`` - Maximum `width * height` of a pixel cache before writing to
      disk.
    - ``'disk'`` - Maximum bytes used by pixel cache on disk before exception
      is thrown.
    - ``'file'`` - Maximum cache files opened at any given time.
    - ``'height'`` - Maximum height of image before exception is thrown.
    - ``'list_length'`` - Maximum images in sequence. Only available with
      recent version of ImageMagick.
    - ``'map'`` - Maximum memory map in bytes to allocated for pixel cache
      before using disk.
    - ``'memory'`` - Maximum bytes to allocated for pixel cache before using
      disk.
    - ``'thread'`` - Maximum parallel task sub-routines can spawn - if using
      OpenMP.
    - ``'throttle'`` - Total milliseconds to yield to CPU - if possible.
    - ``'time'`` - Maximum seconds before exception is thrown.
    - ``'width'`` - Maximum width of image before exception is thrown.

    .. versionadded:: 0.5.1
    """

    #: (:class:`tuple`) List of available resource types for ImageMagick-6.
    _limits6 = ('undefined', 'area', 'disk', 'file', 'map', 'memory', 'thread',
                'time', 'throttle', 'width', 'height')

    #: (:class:`tuple`) List of available resource types for ImageMagick-7.
    _limits7 = ('undefined', 'area', 'disk', 'file', 'height', 'map', 'memory',
                'thread', 'throttle', 'time', 'width', 'list_length')

    def __init__(self):
        if MAGICK_VERSION_NUMBER < 0x700:
            self.limits = self._limits6
        else:
            self.limits = self._limits7

    def __getitem__(self, r):
        return self.get_resource_limit(r)

    def __setitem__(self, r, v):
        self.set_resource_limit(r, v)

    def __delitem__(self, r):
        self[r] = 0

    def __iter__(self):
        return iter(self.limits)

    def __len__(self):
        return len(self.limits)

    def _to_idx(self, resource):
        """Helper method to map resource string to enum value."""
        return self.limits.index(resource)

    def resource(self, resource):
        """Get the current value for the resource type.

        :param resource: Resource type.
        :type resource: :class:`basestring`
        :rtype: :class:`numeric.Integral`

        .. versionadded:: 0.5.1
        """
        return libmagick.GetMagickResource(self._to_idx(resource))

    def get_resource_limit(self, resource):
        """Get the current limit for the resource type.

        :param resource: Resource type.
        :type resource: :class:`basestring`
        :rtype: :class:`numeric.Integral`

        .. versionadded:: 0.5.1
        """
        return libmagick.GetMagickResourceLimit(self._to_idx(resource))

    def set_resource_limit(self, resource, limit):
        """Sets a new limit for resource type.

        .. note::

            The new limit value must be equal to, or less then, the maximum
            limit defined by the :file:`policy.xml`. Any values set outside
            normal bounds will be ignored silently.

        :param resource: Resource type.
        :type resource: :class:`basestring`
        :param limit: New limit value.
        :type limit: :class:`numeric.Integral`

        .. versionadded:: 0.5.1
        """
        libmagick.SetMagickResourceLimit(self._to_idx(resource), int(limit))


#: (:class:`ResourceLimits`) Helper to get & set Magick Resource Limits.
#:
#: .. versionadded:: 0.5.1
limits = ResourceLimits()
