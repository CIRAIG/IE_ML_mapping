"""
This Python module uses the code from https://github.com/UKPLab/sentence-transformers to match lists of words to
existing classifications in the world of Industrial Ecology.

author: maxime.agez@polymtl.ca
"""

import pandas as pd
import json
import pkg_resources
from sentence_transformers import (SentenceTransformer, util)


class Mapping:
    def __init__(self, reference_classification, transformer_model, number_of_guesses):
        """
        :param reference_classification: [string] The reference classification that is used for matching.
                Available choices:
                    - IOCC
                    - openIO-Canada
                    - NACE
                    - exiobase
                    - IMPACT World+
        :param transformer_model: [string] The name of the machine learning model to be used for matching. THe different
                                  available models are described here: https://www.sbert.net/docs/pretrained_models.html
        :param number_of_guesses: [integer] The amount of suggestions made by the ML algorithm that will be displayed.
        """

        self.reference_classification = reference_classification
        self.model = SentenceTransformer(transformer_model)
        self.number_of_guesses = number_of_guesses

        # define attributes
        self.mapping = None
        self.sorted_scores = None
        self.indices = None
        self.inputs = None
        self.input_embeddings = None
        self.reference_embeddings = None
        self.iocc_sectors = None
        self.nace_sectors = None
        self.exio_sectors = None
        self.iw_flows = None

        if self.reference_classification in ['IOCC', 'openIO-Canada']:
            self.match_to_iocc()
        elif self.reference_classification in ['NACE']:
            self.match_to_nace()
        elif self.reference_classification in ['exiobase']:
            self.match_to_exio()
        elif self.reference_classification in ['IMPACT World+']:
            self.match_to_iw()

    def match_inputs(self, inputs):
        """
        Loads the list of inputs to the machine learning model.
        :param inputs: a [list] of words to-be-matched with the reference classification
        :return:
        """
        self.inputs = inputs
        self.input_embeddings = self.model.encode(self.inputs)

    def match_to_iocc(self):
        """
        Method maps list of inputs to the Detailed level of the IOCC classification, notably used in the Input-Output
        database OpenIO-Canada.
        :return:
        """
        with open(pkg_resources.resource_filename(__name__, '/Data/IOCC_sectors.json'), 'r') as f:
            self.iocc_sectors = json.load(f)
        self.reference_embeddings = self.model.encode(self.iocc_sectors)

    def match_to_nace(self):
        """
        Method maps list of inputs to the NACE classification.
        :return:
        """
        with open(pkg_resources.resource_filename(__name__, '/Data/NACE_sectors.json'), 'r') as f:
            self.nace_sectors = json.load(f)
        self.reference_embeddings = self.model.encode([i[1] for i in self.nace_sectors])

    def match_to_exio(self):
        """
        Method maps list of inputs to the exiobase classification, the latter is inspired by NACE.
        :return:
        """
        with open(pkg_resources.resource_filename(__name__, '/Data/exiobase_sectors.json'), 'r') as f:
            self.exio_sectors = json.load(f)
        self.reference_embeddings = self.model.encode(self.exio_sectors)

    def match_to_iw(self):
        """
        Method maps list of inputs to th classification used by the IMPACT World+ LCIA methodology.
        :return:
        """
        with open(pkg_resources.resource_filename(__name__, '/Data/IW_2.0_flows.json'), 'r') as f:
            self.iw_flows = json.load(f)
        self.reference_embeddings = self.model.encode(self.iw_flows)

    def calculate_scores(self):
        """
        Calculates similarity scores.
        :return: a sorted list of the similarity scores and the associated indices
        """

        scores = util.pytorch_cos_sim(self.input_embeddings, self.reference_embeddings)
        self.sorted_scores, self.indices = scores.sort(dim=1, descending=True)

    def format_results(self):
        """
        Formats the results in a dataframe.
        :return: self.mapping, the final mapping that the user is after
        """

        if self.reference_classification in ['IOCC', 'openIO-Canada']:
            reference_list = self.iocc_sectors
        elif self.reference_classification in ['NACE']:
            reference_list = self.nace_sectors
        elif self.reference_classification in ['exiobase']:
            reference_list = self.exio_sectors
        elif self.reference_classification in ['IMPACT World+']:
            reference_list = self.iw_flows

        if self.reference_classification in ['IOCC', 'openIO-Canada', 'exiobase', 'IMPACT World+']:

            self.mapping = pd.DataFrame(None, ['order', 'sector', 'similarity'])
            for i, product in enumerate(self.inputs):
                for j in range(0, self.number_of_guesses):
                    self.mapping = pd.concat([self.mapping,
                                              pd.DataFrame([product,
                                                            j + 1,
                                                            reference_list[self.indices[i][j].cpu().numpy()],
                                                            self.sorted_scores[i][j].cpu().numpy().tolist()],
                                                           ['product', 'order', 'sector', 'similarity'])],
                                             axis=1)
            self.mapping = self.mapping.T.set_index(['product', 'order'])
            return self.mapping

        elif self.reference_classification in ['NACE']:

            self.mapping = pd.DataFrame(None, ['order', 'sector', 'similarity'])
            for i, product in enumerate(self.inputs):
                for j in range(0, self.number_of_guesses):
                    self.mapping = pd.concat([self.mapping,
                                              pd.DataFrame([product,
                                                            j + 1,
                                                            reference_list[self.indices[i][j].cpu().numpy()][0],
                                                            reference_list[self.indices[i][j].cpu().numpy()][1],
                                                            self.sorted_scores[i][j].cpu().numpy().tolist()],
                                                           ['product', 'order', 'code sector', 'sector', 'similarity'])],
                                             axis=1)
            self.mapping = self.mapping.T.set_index(['product', 'order'])
            self.mapping = self.mapping.T.reindex(['code sector', 'sector', 'similarity']).T
            return self.mapping
