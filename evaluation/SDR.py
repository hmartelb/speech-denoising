import numpy as np

class SDR():
    '''
    Estimation of SDR:
        true_sources_list: 3d dim (num_samples, num_channels, num_sources)
        estimated_sources_list: 3d dim (num_samples, num_channels, num_sources)
    References:
        https://arxiv.org/pdf/1811.02508.pdf
        https://github.com/sigsep/bsseval/issues/3
        https://github.com/sigsep/bsseval/blob/master/bsseval/__init__.py
    '''

    def __init__(self, true_sources_list, estimated_sources_list, scaling=True):
        self.reference_array = true_sources_list
        self.estimated_array = estimated_sources_list
        self.scaling = scaling

    def evaluate(self):
        num_sources = self.reference_array.shape[-1]
        num_channels = self.reference_array.shape[1]
        orderings = ( [list(range(num_sources))] )
        results = np.empty((len(orderings), num_channels, num_sources, 3))

        for o, order in enumerate(orderings):
            for c in range(num_channels):
                for j in order:
                    SDR = self._compute_sdr(
                        self.estimated_array[:, c, j], self.reference_array[:, c, order], j, scaling=self.scaling
                    )
                    results[o, c, j, :] = [SDR]
        return results

    @staticmethod
    def _compute_sdr(estimated_signal, reference_signals, source_idx, scaling=True):
        references_projection = reference_signals.T @ reference_signals
        source = reference_signals[:, source_idx]
        scale = (source @ estimated_signal) / references_projection[source_idx, source_idx] if scaling else 1

        e_true = scale * source
        e_res = estimated_signal - e_true

        signal = (e_true ** 2).sum()
        noise = (e_res ** 2).sum()
        SDR = 10 * np.log10(signal / noise)

        return SDR

if __name__ == '__main__':
    #given as an input to the model
    true_src_list = ''
    #output of the model
    estimated_src_list = ''
    sdr = SDR(true_src_list, estimated_src_list)
    print(sdr.evaluate())
