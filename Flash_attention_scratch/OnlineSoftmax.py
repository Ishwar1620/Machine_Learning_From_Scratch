class Onlinesoftmax():
    def __init__(self):
        self.max = float('-inf')
        self.sum_exp = 0
        self.prob = []

    def update(self,new_value):
        import math
        if not isinstance(new_value,list):
            new_value = [new_value]
        old_max = self.max
        old_sum_exp = self.sum_exp
        self.max = max(self.max,*new_value)
        rescale_factor = math.exp(old_max - self.max)
        self.sum_exp *= rescale_factor

        for value in new_value:
            self.sum_exp += math.exp(value - self.max)

        if len(self.prob) != 0:
            for i in range(len(self.prob)):
                self.prob[i] = self.prob[i] *rescale_factor* old_sum_exp / self.sum_exp
            for value in new_value:
                new_prob = math.exp(value - self.max)/self.sum_exp
                self.prob.append(new_prob)
        else:
            for value in new_value:
                new_prob = math.exp(value - self.max)/self.sum_exp
                self.prob.append(new_prob)
        
def main():
    # Test 1: Single batch (should match regular softmax)
    softmax = Onlinesoftmax()
    softmax.update([1, 2, 3])
    print("Single batch:", softmax.prob)

    # Test 2: Incremental updates
    softmax2 = Onlinesoftmax()
    softmax2.update([1, 2])
    print("After [1, 2]:", softmax2.prob)
    softmax2.update([3])
    print("After adding [3]:", softmax2.prob)

    # Test 3: Compare with reference
    import math
    def reference_softmax(values):
        max_val = max(values)
        exp_vals = [math.exp(v - max_val) for v in values]
        sum_exp = sum(exp_vals)
        return [e / sum_exp for e in exp_vals]

    print("Reference [1,2,3]:", reference_softmax([1, 2, 3]))
main()