
class ItemData():
    """
    This class represents item data as read in from a file created by preprocessing.
    The file is in CSV form and has the following columns:
    - item_id: unique integer
    - provider_id: the id of the associated provider. Must match a provider id in the provider data file
    - features: a [] delimited, space-separated vector of features, which map on the vectors for user
        preferences
    """