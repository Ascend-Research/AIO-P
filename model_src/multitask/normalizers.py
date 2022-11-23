import numpy as np

class MultiTaskNormalizer:
    def __init__(self, data=None, type="stand"):
        self.type = type
        self.subtractor, self.divisor = None, None
        if data is not None:
            self.generate_params(data)

    def generate_params(self, norm_data):
        tar_list = self._make_cg_target_list(norm_data)
        if "flops" in self.type:
            print("Using FLOPs!")
            tar_list = self._apply_flops_quotient(tar_list, self._make_cg_target_list(norm_data, idx=1))
        if "stand" in self.type:
            self.subtractor = np.mean(tar_list)
            self.divisor = np.std(tar_list)
        
        else:
            print("UNSPECIFIED TRANSFORM GIVEN. DOING NOTHING!")
            self.subtractor, self.divisor = 0, 1

    @staticmethod
    def _make_cg_target_list(norm_data, idx=-1):
        # Assume norm_data is a list of [CG, target] pairs
        return np.array([data[idx] for data in norm_data])

    @staticmethod
    def _flops_transform(flops):
        return np.log10(flops + 1) + 1

    @staticmethod
    def _apply_flops_quotient(acc, flops):
        log_flops = MultiTaskNormalizer._flops_transform(flops)
        return acc / log_flops

    def transform(self, data):
        targets = self._make_cg_target_list(data)
        if "flops" in self.type:
            targets = self._apply_flops_quotient(targets, self._make_cg_target_list(data, idx=1))
        targets = (targets - self.subtractor) / self.divisor
        targets = targets.tolist()
        return self._reassign(data, targets)

    def inverse(self, predictions, flops):
        predictions = (predictions * self.divisor) + self.subtractor
        if "flops" in self.type:
            predictions *= self._flops_transform(flops)
        return predictions

    @staticmethod
    def _reassign(data, new_tars):
        for i in range(len(new_tars)):
            data[i][-1] = new_tars[i]
        return data
