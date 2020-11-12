try:
    from elixer.global_config import __version__ as elixer_version
except:
    import global_config.__version__ as elixer_version
__version__ = elixer_version
