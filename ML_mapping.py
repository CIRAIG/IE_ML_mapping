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
                    - openIO-Canada
                    - exiobase
                    - USEEIO 2.0
                    - GTAP 10
                    - IOCC
                    - NACE Rev.1.1
                    - NACE Rev.2
                    - CPA 2008
                    - CPA 2.1
                    - NAPCS 2017
                    - NAPCS 2022
                    - NAICS 2017
                    - NAICS 2022
                    - ISIC Rev.4
                    - CPC 2.1
                    - COICOP 2018
                    - ecoinvent 3.8 technosphere
                    - ecoinvent 3.9 technosphere
                    - ecoinvent 3.8 elementary flows
                    - ecoinvent 3.9 elementary flows
                    - IMPACT World+ 2.0
                    - USEtox 2
                    - EF 3.0
                    - EF 3.1
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
        self.exio_sectors = None
        self.useeio_sectors = None
        self.gtap_sectors = None
        self.nace_1_1_sectors = None
        self.nace_2_sectors = None
        self.cpa_2008_sectors = None
        self.cpa_2_1_sectors = None
        self.napcs_2017_sectors = None
        self.napcs_2022_sectors = None
        self.naics_2017_sectors = None
        self.naics_2022_sectors = None
        self.isic_4_sectors = None
        self.cpc_2_1_sectors = None
        self.coicop_sectors = None
        self.ecoinvent_3_8_technosphere = None
        self.ecoinvent_3_9_technosphere = None
        self.ecoinvent_3_8_flows = None
        self.ecoinvent_3_9_flows = None
        self.iw_flows = None
        self.usetox_flows = None
        self.ef_3_0_flows = None
        self.ef_3_1_flows = None

        if self.reference_classification in ['IOCC', 'openIO-Canada']:
            self.match_to_iocc()
        elif self.reference_classification in ['NACE Rev.1.1']:
            self.match_to_nace_1_1()
        elif self.reference_classification in ['NACE Rev.2']:
            self.match_to_nace_2()
        elif self.reference_classification in ['CPA 2008']:
            self.match_to_cpa_2008()
        elif self.reference_classification in ['CPA 2.1']:
            self.match_to_cpa_2_1()
        elif self.reference_classification in ['exiobase']:
            self.match_to_exio()
        elif self.reference_classification in ['USEEIO 2.0']:
            self.match_to_useeio()
        elif self.reference_classification in ['GTAP 10']:
            self.match_to_gtap()
        elif self.reference_classification in ['NAPCS 2017']:
            self.match_to_napcs_2017()
        elif self.reference_classification in ['NAPCS 2022']:
            self.match_to_napcs_2022()
        elif self.reference_classification in ['NAICS 2017']:
            self.match_to_naics_2017()
        elif self.reference_classification in ['NAICS 2022']:
            self.match_to_naics_2022()
        elif self.reference_classification in ['ISIC Rev.4']:
            self.match_to_isic_4()
        elif self.reference_classification in ['CPC 2.1']:
            self.match_to_cpc_2_1()
        elif self.reference_classification in ['COICOP 2018']:
            self.match_to_coicop()
        elif self.reference_classification in ['ecoinvent 3.8 technosphere']:
            self.match_to_ei38_techno()
        elif self.reference_classification in ['ecoinvent 3.9 technosphere']:
            self.match_to_ei39_techno()
        elif self.reference_classification in ['ecoinvent 3.8 elementary flows']:
            self.match_to_ei38_flows()
        elif self.reference_classification in ['ecoinvent 3.9 elementary flows']:
            self.match_to_ei39_flows()
        elif self.reference_classification in ['IMPACT World+ 2.0']:
            self.match_to_iw()
        elif self.reference_classification in ['USEtox 2']:
            self.match_to_usetox()
        elif self.reference_classification in ['EF 3.0']:
            self.match_to_ef_3_0()
        elif self.reference_classification in ['EF 3.1']:
            self.match_to_ef_3_1()

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

    def match_to_nace_1_1(self):
        """
        Method maps list of inputs to the NACE Rev. 1.1 classification.
        :return:
        """
        with open(pkg_resources.resource_filename(__name__, '/Data/NACE_1_1_sectors.json'), 'r') as f:
            self.nace_1_1_sectors = json.load(f)
        self.reference_embeddings = self.model.encode([i[1] for i in self.nace_1_1_sectors])

    def match_to_nace_2(self):
        """
        Method maps list of inputs to the NACE Rev. 2 classification.
        :return:
        """
        with open(pkg_resources.resource_filename(__name__, '/Data/NACE_2_sectors.json'), 'r') as f:
            self.nace_2_sectors = json.load(f)
        self.reference_embeddings = self.model.encode([i[1] for i in self.nace_2_sectors])

    def match_to_cpa_2008(self):
        """
        Method maps list of inputs to the CPA 2008 classification.
        :return:
        """
        with open(pkg_resources.resource_filename(__name__, '/Data/CPA_2008_sectors.json'), 'r') as f:
            self.cpa_2008_sectors = json.load(f)
        self.reference_embeddings = self.model.encode([i[1] for i in self.cpa_2008_sectors])

    def match_to_cpa_2_1(self):
        """
        Method maps list of inputs to the CPA 2.1 classification.
        :return:
        """
        with open(pkg_resources.resource_filename(__name__, '/Data/CPA_2_1_sectors.json'), 'r') as f:
            self.cpa_2_1_sectors = json.load(f)
        self.reference_embeddings = self.model.encode([i[1] for i in self.cpa_2_1_sectors])

    def match_to_exio(self):
        """
        Method maps list of inputs to the exiobase classification, the latter is inspired by NACE.
        :return:
        """
        with open(pkg_resources.resource_filename(__name__, '/Data/exiobase_sectors.json'), 'r') as f:
            self.exio_sectors = json.load(f)
        self.reference_embeddings = self.model.encode(self.exio_sectors)

    def match_to_useeio(self):
        """
        Method maps list of inputs to the USEEIO 2.0 classification.
        :return:
        """
        with open(pkg_resources.resource_filename(__name__, '/Data/USEEIO_sectors.json'), 'r') as f:
            self.useeio_sectors = json.load(f)
        self.reference_embeddings = self.model.encode(self.useeio_sectors)

    def match_to_gtap(self):
        """
        Method maps list of inputs to the classification of the GTAP database.
        :return:
        """
        with open(pkg_resources.resource_filename(__name__, '/Data/GTAP_sectors.json'), 'r') as f:
            self.gtap_sectors = json.load(f)
        self.reference_embeddings = self.model.encode(self.gtap_sectors)

    def match_to_napcs_2017(self):
        """
        Method maps list of inputs to the NAPCS 2017 classification.
        :return:
        """
        with open(pkg_resources.resource_filename(__name__, '/Data/NAPCS_2017_sectors.json'), 'r') as f:
            self.napcs_2017_sectors = json.load(f)
        self.reference_embeddings = self.model.encode([i[1] for i in self.napcs_2017_sectors])

    def match_to_napcs_2022(self):
        """
        Method maps list of inputs to the NAPCS 2022 classification.
        :return:
        """
        with open(pkg_resources.resource_filename(__name__, '/Data/NAPCS_2022_sectors.json'), 'r') as f:
            self.napcs_2022_sectors = json.load(f)
        self.reference_embeddings = self.model.encode([i[1] for i in self.napcs_2022_sectors])

    def match_to_naics_2017(self):
        """
        Method maps list of inputs to the NAICS 2017 classification.
        :return:
        """
        with open(pkg_resources.resource_filename(__name__, '/Data/NAICS_2017_sectors.json'), 'r') as f:
            self.naics_2017_sectors = json.load(f)
        self.reference_embeddings = self.model.encode([i[1] for i in self.naics_2017_sectors])

    def match_to_naics_2022(self):
        """
        Method maps list of inputs to the NAICS 2022 classification.
        :return:
        """
        with open(pkg_resources.resource_filename(__name__, '/Data/NAICS_2022_sectors.json'), 'r') as f:
            self.naics_2022_sectors = json.load(f)
        self.reference_embeddings = self.model.encode([i[1] for i in self.naics_2022_sectors])

    def match_to_isic_4(self):
        """
        Method maps list of inputs to the ISIC Rev.4 classification.
        :return:
        """
        with open(pkg_resources.resource_filename(__name__, '/Data/ISIC_4_sectors.json'), 'r') as f:
            self.isic_4_sectors = json.load(f)
        self.reference_embeddings = self.model.encode([i[1] for i in self.isic_4_sectors])

    def match_to_cpc_2_1(self):
        """
        Method maps list of inputs to the CPC 2.1 classification.
        :return:
        """
        with open(pkg_resources.resource_filename(__name__, '/Data/CPC_2_1_sectors.json'), 'r') as f:
            self.cpc_2_1_sectors = json.load(f)
        self.reference_embeddings = self.model.encode([i[1] for i in self.cpc_2_1_sectors])

    def match_to_coicop(self):
        """
        Method maps list of inputs to the COICOP classification.
        :return:
        """
        with open(pkg_resources.resource_filename(__name__, '/Data/COICOP_2018_sectors.json'), 'r') as f:
            self.coicop_sectors = json.load(f)
        self.reference_embeddings = self.model.encode([i[1] for i in self.coicop_sectors])

    def match_to_iw(self):
        """
        Method maps list of inputs to the classification used by the IMPACT World+ LCIA methodology.
        :return:
        """
        with open(pkg_resources.resource_filename(__name__, '/Data/IW_2.0_flows.json'), 'r') as f:
            self.iw_flows = json.load(f)
        self.reference_embeddings = self.model.encode(self.iw_flows)

    def match_to_ei38_techno(self):
        """
        Method maps list of inputs to the product classification used by the ecoinvent 3.8 database.
        :return:
        """
        with open(pkg_resources.resource_filename(__name__, '/Data/ecoinvent_3_8_sectors.json'), 'r') as f:
            self.ecoinvent_3_8_technosphere = json.load(f)
        self.reference_embeddings = self.model.encode(self.ecoinvent_3_8_technosphere)

    def match_to_ei39_techno(self):
        """
        Method maps list of inputs to the product classification used by the ecoinvent 3.9 database.
        :return:
        """
        with open(pkg_resources.resource_filename(__name__, '/Data/ecoinvent_3_9_sectors.json'), 'r') as f:
            self.ecoinvent_3_9_technosphere = json.load(f)
        self.reference_embeddings = self.model.encode(self.ecoinvent_3_9_technosphere)

    def match_to_ei38_flows(self):
        """
        Method maps list of inputs to the elementary flow classification used by the ecoinvent 3.8 database.
        :return:
        """
        with open(pkg_resources.resource_filename(__name__, '/Data/ecoinvent_3_8_flows.json'), 'r') as f:
            self.ecoinvent_3_8_flows = json.load(f)
        self.reference_embeddings = self.model.encode(self.ecoinvent_3_8_flows)

    def match_to_ei39_flows(self):
        """
        Method maps list of inputs to the elementary flow classification used by the ecoinvent 3.9 database.
        :return:
        """
        with open(pkg_resources.resource_filename(__name__, '/Data/ecoinvent_3_9_flows.json'), 'r') as f:
            self.ecoinvent_3_9_flows = json.load(f)
        self.reference_embeddings = self.model.encode(self.ecoinvent_3_9_flows)

    def match_to_usetox(self):
        """
        Method maps list of inputs to the classification used by the USEtox database.
        :return:
        """
        with open(pkg_resources.resource_filename(__name__, '/Data/USEtox_flows.json'), 'r') as f:
            self.usetox_flows = json.load(f)
        self.reference_embeddings = self.model.encode(self.usetox_flows)

    def match_to_ef_3_0(self):
        """
        Method maps list of inputs to the classification used by the EF 3.0 LCIA method.
        :return:
        """
        with open(pkg_resources.resource_filename(__name__, '/Data/EF_3_0_flows.json'), 'r') as f:
            self.ef_3_0_flows = json.load(f)
        self.reference_embeddings = self.model.encode(self.ef_3_0_flows)

    def match_to_ef_3_1(self):
        """
        Method maps list of inputs to the classification used by the EF 3.1 LCIA method.
        :return:
        """
        with open(pkg_resources.resource_filename(__name__, '/Data/EF_3_1_flows.json'), 'r') as f:
            self.ef_3_1_flows = json.load(f)
        self.reference_embeddings = self.model.encode(self.ef_3_1_flows)

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
        elif self.reference_classification in ['NACE Rev.1.1']:
            reference_list = self.nace_1_1_sectors
        elif self.reference_classification in ['NACE Rev.2']:
            reference_list = self.nace_2_sectors
        elif self.reference_classification in ['CPA 2008']:
            reference_list = self.cpa_2008_sectors
        elif self.reference_classification in ['CPA 2.1']:
            reference_list = self.cpa_2_1_sectors
        elif self.reference_classification in ['exiobase']:
            reference_list = self.exio_sectors
        elif self.reference_classification in ['USEEIO 2.0']:
            reference_list = self.useeio_sectors
        elif self.reference_classification in ['GTAP 10']:
            reference_list = self.gtap_sectors
        elif self.reference_classification in ['NAPCS 2017']:
            reference_list = self.napcs_2017_sectors
        elif self.reference_classification in ['NAPCS 2022']:
            reference_list = self.napcs_2022_sectors
        elif self.reference_classification in ['NAICS 2017']:
            reference_list = self.naics_2017_sectors
        elif self.reference_classification in ['NAICS 2022']:
            reference_list = self.naics_2022_sectors
        elif self.reference_classification in ['ISIC Rev.4']:
            reference_list = self.isic_4_sectors
        elif self.reference_classification in ['CPC 2.1']:
            reference_list = self.cpc_2_1_sectors
        elif self.reference_classification in ['COICOP 2018']:
            reference_list = self.coicop_sectors
        elif self.reference_classification in ['ecoinvent 3.8 technosphere']:
            reference_list = self.ecoinvent_3_8_technosphere
        elif self.reference_classification in ['ecoinvent 3.9 technosphere']:
            reference_list = self.ecoinvent_3_9_technosphere
        elif self.reference_classification in ['ecoinvent 3.8 elementary flows']:
            reference_list = self.ecoinvent_3_8_flows
        elif self.reference_classification in ['ecoinvent 3.9 elementary flows']:
            reference_list = self.ecoinvent_3_9_flows
        elif self.reference_classification in ['IMPACT World+ 2.0']:
            reference_list = self.iw_flows
        elif self.reference_classification in ['USEtox 2']:
            reference_list = self.usetox_flows
        elif self.reference_classification in ['EF 3.0']:
            reference_list = self.ef_3_0_flows
        elif self.reference_classification in ['EF 3.1']:
            reference_list = self.ef_3_1_flows

        if self.reference_classification in ['IOCC', 'openIO-Canada', 'exiobase', 'USEEIO 2.0', 'GTAP 10',
                                             'ecoinvent 3.8 elementary flows',  'ecoinvent 3.9 elementary flows',
                                             'ecoinvent 3.8 technosphere', 'ecoinvent 3.9 technosphere',
                                             'IMPACT World+ 2.0', 'USEtox 2', 'EF 3.0', 'EF 3.1']:

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

        elif self.reference_classification in ['NACE Rev.1.1', 'NACE Rev.2', 'CPA 2008', 'CPA 2.1', 'NAPCS 2017',
                                               'NAPCS 2022', 'NAICS 2017', 'NAICS 2022', 'ISIC Rev.4', 'CPC 2.1',
                                               'COICOP 2018']:

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
