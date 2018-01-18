# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import json


class Dataset(object):
    """Storage container for a coordinate- and/or image-based meta-analytic
    dataset/database.
    """
    def __init__(self, dataset_file):
        with open(dataset_file, 'r') as fo:
            self.data = json.load(fo)

    def has_data(self, dat_str):
        dat_str = dat_str.split(' AND ')
        for ds in dat_str:
            try:
                self.data.get(None)
            except:
                raise Exception('Nope')

    def get(self, search='', algorithm=None):
        """Retrieve files and/or metadata from the current Dataset.
        """
        if algorithm:
            req_data = algorithm.req_data
            temp = [stud for stud in self.data if stud.has_data(req_data)]

    def get_studies(self):
        pass

    def get_metadata(self):
        pass

    def get_images(self):
        pass

    def get_coordinates(self):
        pass
