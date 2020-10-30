class BlindClassifier():
    """Blind a classifier to user-specified columns

    The blind classifier substitutes random values from the background 
    dataset into the blinded columns before sending the data to the 
    original classifier for prediction. It repeats this `nsamples` times, 
    then outputs the modal prediction.

    Parameters
    ----------
    model : callable
        Callable which takes a (# samples x # features) matrix and returns 
        a (# samples x 1) output vector.

    data : numpy.array, pandas.DataFrame, or pandas.Series
        Background dataset from which variables are randomly sampled to 
        blind the `model` to user-specified columns.

    columns : list of column names or indicies
        Columns to which the classifier is blinded.

    nsamples : scalar
        Number of random values sampled to produce the output.
    """
    def __init__(self, model, data, columns, nsamples=1):
        self.model = model
        if isinstance(data, pd.Series):
            self._from_pandas(data, list(data.index), columns)
        elif isinstance(data, pd.DataFrame):
            self._from_pandas(data, list(data.columns), columns)
        else:
            self.data = data
            self.columns = columns
        self.data = self._reshape_to_2d(self.data)
        # Boolean mask for blinded columns
        self.blinding_mask = np.zeros(self.data.shape[1])
        self.blinding_mask[self.columns] = 1
        # Boolean mask for visible columns
        self.visible_mask = np.logical_not(self.blinding_mask).astype(int)
        self.nsamples = nsamples

    def _from_pandas(self, data, columns, blind_columns):
        # Get data and columns from pandas DataFrame or Series
        self.data = data.values
        self.columns = [columns.index(c) for c in blind_columns]

    def _reshape_to_2d(self, X):
        # Reshape a (px1) vector to a (1xp) rector
        return X.reshape(1, X.shape[0]) if len(X.shape) == 1 else X

    def predict(self, X):
        # Classification prediction
        X = X.values if isinstance(X, (pd.Series, pd.DataFrame)) else X
        X = self._reshape_to_2d(X)
        assert X.shape[1] == self.data.shape[1]
        output = np.array(
            [self._compute_output(X) for i in range(self.nsamples)]
        )
        return scipy.stats.mode(output)[0][0]

    def _compute_output(self, X):
        # Substitute random values from the background dataset into the 
        # blinded coluns
        Z = np.array(choices(self.data, k=X.shape[0]))
        X_blind = self.visible_mask * X + self.blinding_mask * Z
        return self.model(X_blind)