class Transform(object):
    def pdf(self, *args, **kwargs):
        raise NotImplementedError("Must be subclassed.")

    def get_standard_params(self, *args, **kwargs):
        raise NotImplementedError("Must be subclassed.")
