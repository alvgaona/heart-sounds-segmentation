function falsePositiveRate(obj)
    obj.FPR = obj.FP/(obj.FP+obj.TN);
end