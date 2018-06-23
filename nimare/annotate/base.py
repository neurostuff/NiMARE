"""
Base classes for topic model-based annotation.
"""
import gzip
import pickle


class AnnotationModel(object):
    """
    Base class for topic and vector models.
    """
    def __init__(self):
        pass

    def save(self, filename, compress=True):
        """
        Pickle the AnnotationModel instance to the provided file.

        Parameters
        ----------
        filename : :obj:`str`
            File to which model will be saved.
        compress : :obj:`bool`, optional
            If True, the file will be compressed with gzip. Otherwise, the
            uncompressed version will be saved. Default = True.
        """
        if compress:
            with gzip.GzipFile(filename, 'wb') as file_object:
                pickle.dump(self, file_object)
        else:
            with open(filename, 'wb') as file_object:
                pickle.dump(self, file_object)

    @classmethod
    def load(cls, filename, compressed=True):
        """
        Load a pickled AnnotationModel instance from file.

        Parameters
        ----------
        filename : :obj:`str`
            Name of file containing model.
        compressed : :obj:`bool`, optional
            If True, the file is assumed to be compressed and gzip will be used
            to load it. Otherwise, it will assume that the file is not
            compressed. Default = True.

        Returns
        -------
        model : :obj:`nimare.annotate.topic.AnnotationModel`
            Loaded model object.
        """
        if compressed:
            try:
                with gzip.GzipFile(filename, 'rb') as file_object:
                    model = pickle.load(file_object)
            except UnicodeDecodeError:
                # Need to try this for python3
                with gzip.GzipFile(filename, 'rb') as file_object:
                    model = pickle.load(file_object, encoding='latin')
        else:
            try:
                with open(filename, 'rb') as file_object:
                    model = pickle.load(file_object)
            except UnicodeDecodeError:
                # Need to try this for python3
                with open(filename, 'rb') as file_object:
                    model = pickle.load(file_object, encoding='latin')

        if not isinstance(model, AnnotationModel):
            raise IOError('Pickled object must be '
                          '`nimare.annotate.topic.AnnotationModel`, '
                          'not {0}'.format(type(model)))

        return model
