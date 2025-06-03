import csv


class ProviderData:
    """
    This class represents provider data as read in from a file created by preprocessing.
    The file is in CSV form and has the following columns:
    - provider_id: unique integer
    - provider_type: the type of the associated provider. Must be a type name known to the ProviderFactor
    - recommender association: "all", "none", or a []-delimited space-separated list of recommender names
    """

    def __init__(self, readable):
        self.reader = csv.reader(readable)

    def __iter__(self):
        return self

    def __next__(self):
        """

        Returns: provider_id (int), provider_type (string), recommender association: a list (empty for "none"),
        recommender names if provided (no error checking), or a list containing '*' for all
        (which is admittedly a strange convention)
        """
        line = []
        self.skip_blank()
        provider_id_str, provider_type, recommenders = line
        provider_type = provider_type.strip()
        recommenders = recommenders.strip()
        provider_id = int(provider_id_str)
        if recommenders == 'none':
            rec_list = []
        elif recommenders[0] == '[':
            rec_list = recommenders[1:-1].split(' ')
        elif recommenders == 'all':
            rec_list = ['*']
        else:
            raise BadRecommenderListError(recommenders)

        return provider_id, provider_type, rec_list
    
    def skip_blank(self):
        # wishing for a do-while loop here
        while True:
            line = next(self.reader)
            if len(line) > 0:
                break


class BadRecommenderListError(Exception):
    def __init__(self, rec_spec):
        self.message = self.message = f'Cannot load ProviderData: Recommender list {rec_spec} is not recognized.'
        super().__init__(self.message)
